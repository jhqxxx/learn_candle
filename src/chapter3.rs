use crate::utils::dataset::{DataLoader, Dataset};
use crate::utils::file::read_txt;
use candle_core::{D, DType, Device, Error, IndexOp, Result, Tensor};
use candle_nn::{
    AdamW, Embedding, Init, Linear, Module, Optimizer, VarBuilder, VarMap, embedding, linear,
    linear_no_bias, loss, ops,
};
use core::f32;
use rand::prelude::SliceRandom;
use std::collections::HashMap;
use tokenizers::tokenizer::Tokenizer;

use crate::impl_dataset;
use crate::utils::net::{plot_loss_curves};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use std::io::Write;

impl_dataset!(TokenDataset);

#[allow(unused)]
pub struct SinusoidalPositonEmbedding {
    pub pos_embedding: Tensor,
}

#[allow(unused)]
impl SinusoidalPositonEmbedding {
    pub fn new(seq_len: usize, hidden_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(hidden_dim % 2, 0, "hidden_dim must be even");
        let mut pos_embedding_vec = Vec::new();
        for pos in 0..seq_len {
            for i in (0..hidden_dim).step_by(2) {
                let pos_i = pos as f32 / 10000.0_f32.powf(i as f32 / hidden_dim as f32);
                let sin = pos_i.sin();
                let cos = pos_i.cos();
                pos_embedding_vec.push(sin);
                pos_embedding_vec.push(cos);
            }
        }
        let pos_embedding = Tensor::from_vec(pos_embedding_vec, (seq_len, hidden_dim), device)?;
        Ok(Self { pos_embedding })
    }
}

#[allow(unused)]
pub struct DotProductAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    d_sqrt: Tensor,
}

#[allow(unused)]
impl DotProductAttention {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, out_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, out_dim, vb.pp("w_v"))?;
        let d_sqrt = 1.0 / (out_dim as f32).sqrt();
        let d_sqrt = Tensor::new(d_sqrt, device)?;
        Ok(Self {
            w_q,
            w_k,
            w_v,
            d_sqrt,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (bs, seq_len, embedding_dim)
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;
        let atten_score = q.matmul(&k.t()?)?; // (bs, seq_len, seq_len)
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, seq_len, embedding_dim)
        Ok(atten_weight)
    }

    pub fn forward_with_mask(&self, x: &Tensor, mask: bool) -> Result<Tensor> {
        let (_, seq_len, _) = x.dims3()?;
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;
        let mut atten_score = q.matmul(&k.t()?)?; // (bs, seq_len, seq_len)
        if mask {
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, x.device())?;
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
            println!("{}", atten_score);
        }
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, seq_len, embedding_dim)
        Ok(atten_weight)
    }
}

#[allow(unused)]
pub struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    out_proj: Linear,
    n_head: usize,
    head_dim: usize,
    out_dim: usize,
    d_sqrt: Tensor,
}

#[allow(unused)]
impl MultiHeadAttention {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        n_head: usize,
        device: &Device,
    ) -> Result<Self> {
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, out_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, out_dim, vb.pp("w_v"))?;
        let out_proj = linear_no_bias(out_dim, out_dim, vb.pp("out_proj"))?;
        let head_dim = out_dim / n_head;
        let d_sqrt = 1.0 / (head_dim as f32).sqrt();
        let d_sqrt = Tensor::new(d_sqrt, device)?.to_dtype(vb.dtype())?;
        Ok(Self {
            w_q,
            w_k,
            w_v,
            out_proj,
            n_head,
            head_dim,
            out_dim,
            d_sqrt,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: bool) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        // (bs, n_head, seq_len, head_dim)
        let q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut atten_score = q.matmul(&k.t()?)?; // (bs, n_head, seq_len, seq_len)
        if mask {
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, x.device())?;
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
            println!("{}", atten_score);
        }
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }

    pub fn forward_with_rope(&self, x: &Tensor, mask: bool) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        // (bs, n_head, seq_len, head_dim)
        let q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let rope = RoPE::new(seq_len, self.head_dim, x.device())?;
        let q = rope.forward(&q)?;
        let k = rope.forward(&k)?;
        let mut atten_score = q.matmul(&k.t()?)?; // (bs, n_head, seq_len, seq_len)
        if mask {
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, x.device())?;
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
            println!("{}", atten_score);
        }
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        // (bs, n_head, seq_len, seq_len)
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }

    pub fn forward_with_buffer(&self, x: &Tensor, buffers: &mut SharedBuffer) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        // (bs, n_head, seq_len, head_dim)
        let q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (mask, rope) = buffers.get(seq_len, self.head_dim, x.device())?;
        let q = rope.forward(&q)?;
        let k = rope.forward(&k)?;
        let mut atten_score = q.matmul(&k.t()?)?; // (bs, n_head, seq_len, seq_len)
        atten_score = mask_filled(&atten_score, mask, f32::NEG_INFINITY)?;
        println!("{}", atten_score);
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let out = self.out_proj.forward(&atten_weight)?;
        Ok(out)
    }
}

// (bs, seq_len, embedding_dim)
// (bs, seq_len, vocab_size)
// 推理时 bs =1
// x-token,(bs, x, embedding_dim)-> (bs, x, vocab_size)  -> 1
// kv cache
// k/v (bs, n_head, x, head_dim)
//  (bs, n_kv_head, x, head_dim)
//
// 1-token->query (bs, n_head, 1, head_dim) key (bs, n_head, 1, head_dim), value:(bs, n_head, 1, head_dim)
// kvcache +1 cat: k/v :(bs, n_head, x+1, head_dim)
// 1-token:  (bs, n_head, 1, head_dim) matmul (bs, n_head, x+1, head_dim).t() -> (bs, n_head, 1, x+1) --atten_score
// ... mutmal v:(bs, n_head, 1, x+1) matmul v- (bs, n_head, x+1, head_dim) -> (bs, n_head, 1, head_dim)
// (bs, 1, vocab_size) -> + 1

#[allow(unused)]
pub struct GroupAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    out_proj: Linear,
    n_head: usize,
    n_kv_head: usize,
    group_size: usize,
    head_dim: usize,
    out_dim: usize,
    d_sqrt: Tensor,
}

#[allow(unused)]
impl GroupAttention {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        device: &Device,
    ) -> Result<Self> {
        assert_eq!(out_dim % n_head, 0, "out_dim must be divisible by n_head");
        assert_eq!(
            n_head % n_kv_head,
            0,
            "n_head must be divisible by n_kv_head"
        );
        let head_dim = out_dim / n_head;
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, n_kv_head * head_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, n_kv_head * head_dim, vb.pp("w_v"))?;
        let out_proj = linear_no_bias(out_dim, out_dim, vb.pp("out_proj"))?;
        let group_size = n_head / n_kv_head;
        let d_sqrt = 1.0 / (head_dim as f32).sqrt();
        let d_sqrt = Tensor::new(d_sqrt, device)?;
        Ok(Self {
            w_q,
            w_k,
            w_v,
            out_proj,
            n_head,
            n_kv_head,
            group_size,
            head_dim,
            out_dim,
            d_sqrt,
        })
    }
    pub fn forward(&self, x: &Tensor, buffers: &mut SharedBuffer) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        let (mask, rope) = buffers.get(seq_len, self.head_dim, x.device())?;
        // (bs, n_head, seq_len, head_dim)
        let q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = rope.forward(&q)?;
        let k = rope.forward(&k)?;
        let k = k.repeat((1, self.group_size, 1, 1))?;
        let v = v.repeat((1, self.group_size, 1, 1))?;
        let mut atten_score = q.matmul(&k.t()?)?;
        atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
        println!("{}", atten_score);
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let atten_weight = self.out_proj.forward(&atten_weight)?;
        Ok(atten_weight)
    }

    pub fn forward_without_mask(&self, x: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        // (bs, n_head, seq_len, head_dim)
        let q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k.repeat((1, self.group_size, 1, 1))?;
        let v = v.repeat((1, self.group_size, 1, 1))?;
        let mut atten_score = q.matmul(&k.t()?)?;
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let atten_weight = self.out_proj.forward(&atten_weight)?;
        Ok(atten_weight)
    }
}

#[allow(unused)]
pub fn bytes_char() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');
    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;

    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }
    // Safety: cs contains all values from bs (between 0 and 255),
    // and some values of value 2⁸ + n, where n is between 0 and 255. This is between 255 and 512.
    // Both ranges are valid UTF-32 values (which is fully saturated until 0xD000)
    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
        .collect()
}

#[allow(unused)]
fn about_tokenize() -> Result<()> {
    // 分词-tokenize
    // 你好啊，吃饭了吗
    // Hello everyone, I'm Fengcheche
    // BPE:Byte Pair Encoding
    // 1字节-8bit 1111 1111-> 0~255
    // 你-三个字节
    let mychar = "你好".as_bytes();
    let len = mychar.len();
    for i in 0..len {
        println!("bytes[{i}]: {}", mychar[i]);
    }
    // 0..len
    // 你好-ä½łå¥½ - 108386
    // 228 189 160 -你 - ä½ł  - 56568
    let chars = bytes_char();
    println!("{:?}", chars);
    let tokenizer = Tokenizer::from_file("assets/tokenizer.json")
        .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    // let vocab_size = tokenizer.get_vocab_size(true);
    // println!("{}", vocab_size);
    let encoding = tokenizer
        .encode("你好啊", true)
        .map_err(|e| Error::Msg(format!("tokenizer encoding error: {}", e)))?;
    let tokens_id = encoding.get_ids();
    println!("{:?}", tokens_id);
    Ok(())
}

#[allow(unused)]
fn about_token_embedding() -> Result<()> {
    // 输入： 文本
    // 实际模型输入是 Tensor
    // 文本 -> Tensor?
    // 你好 - 108386
    // 分词将文本-> token id
    // 嵌入将token id -> Tensor?
    //
    // 0-151668    词表大小为151669
    // 维度为 词表大小*隐藏维度-》可学习矩阵
    // 256~4096
    // 768
    // 151669*768
    let device = Device::cuda_if_available(0)?;
    //tokenizer
    let tokenizer = Tokenizer::from_file("assets/tokenizer.json")
        .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("{}", vocab_size);
    let encoding = tokenizer
        .encode("你好啊, 吃饭了吗", true)
        .map_err(|e| Error::Msg(format!("tokenizer encoding error: {}", e)))?;
    let tokens_id = encoding.get_ids();
    println!("{:?}", tokens_id);
    // 7*768
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let embedding = embedding(vocab_size, 768, vb.pp("embedding"))?;
    let tokend_id_tensor = Tensor::new(tokens_id, &device)?;
    let embeded = embedding.forward(&tokend_id_tensor)?;
    println!("{:?}", embeded);
    Ok(())
}

#[allow(unused)]
fn about_token_dataset() -> Result<()> {
    // inputs-targets
    // 输入-标签
    // 你好啊，你吃了吗
    // [108386, 103924, 11, 38433, 225, 99938, 105660]
    //      [108386, 103924, 11, 38433, 225, 99938]
    //          [103924, 11, 38433, 255, 99938, 105660]
    // 10000*768
    // 1024
    // 256/512 -> seq_len 768+256
    // stride 128
    // 0-256 / 128-384 / 256-512/
    // 4-seq->bs
    // bs, seq_len,
    // bs, seq_len, embedding_dim
    // string, &tokenizer, seq_len, stride
    let device = Device::cuda_if_available(0)?;
    let txt = read_txt("assets/sub_wiki_0_99.txt");
    let tokenizer = Tokenizer::from_file("assets/tokenizer.json")
        .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    let token_dataset = TokenDataset::new(txt, &tokenizer, 32, 32, &device)?;
    let (x, y) = token_dataset.get_batch(0, 1)?;
    println!("{}", x);
    println!("{}", y);
    Ok(())
}

#[allow(unused)]
fn about_position_embedding() -> Result<()> {
    // 位置嵌入
    // 你好，你好可爱啊
    // [108386, 3837, 108386, 102783, 103924]
    // 词嵌入-》 token embedding ->
    // hidden_dim: 256 5*256,0行和2行向量是一样的
    // 注意力计算时
    // w_k, w_q, w_v(hidden_dim, hidden_dim)-> q-(5*256),k-(5*256),v-(5*256)
    // q*k^T   -- attenion
    // 词嵌入 + 位置嵌入-》
    // 可学习的
    // cos sin
    // rope

    let device = Device::cuda_if_available(0)?;
    let tokenizer = Tokenizer::from_file("assets/tokenizer.json")
        .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    let vocab_size = tokenizer.get_vocab_size(true);
    let encode = tokenizer
        .encode("你好，你好可爱啊", true)
        .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    let token_ids = encode.get_ids();
    println!("{:?}", token_ids);

    // // let txt = read_txt("assets/sub_wiki_0_99.txt");
    // // let token_dataset = TokenDataset::new(txt, &tokenizer, 32, 32, &device)?;
    // // let (x, y) = token_dataset.get_batch(0, 1)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let embedding_dim = 8;
    let embedding = embedding(vocab_size, embedding_dim, vb.pp("embedding"))?;
    let token_tensor = Tensor::new(token_ids, &device)?;
    let token_tensor = token_tensor.unsqueeze(0)?;
    let token_tensor = Tensor::cat(&[&token_tensor, &token_tensor], 0)?;
    let encode = embedding.forward(&token_tensor)?;
    println!("encode: {}", encode);
    // let pos_embedding = SinusoidalPositonEmbedding::new(5, embedding_dim, &device)?.pos_embedding;
    // let encode = encode.add(&pos_embedding)?;
    // println!("{}", encode);

    let linear1 = linear(embedding_dim, embedding_dim, vb.pp("w_q"))?;
    let linear2 = linear(embedding_dim, embedding_dim, vb.pp("w_k"))?;
    let linear3 = linear(embedding_dim, embedding_dim, vb.pp("w_v"))?;
    let q = linear1.forward(&encode)?;
    let k = linear2.forward(&encode)?;
    let v = linear3.forward(&encode)?;
    // let rope = RoPE::new(5, embedding_dim, &device)?;
    // let q = rope.forward(&q)?;
    // let k = rope.forward(&k)?;
    let rope = RoPE::new(5, embedding_dim, &device)?;
    let q = rope.forward(&q)?;
    let k = rope.forward(&k)?;
    let atten_score = q.matmul(&k.t()?)?;
    println!("{}", atten_score);
    Ok(())
}

#[allow(unused)]
fn about_attention() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let tokenizer = Tokenizer::from_file("assets/tokenizer.json")
        .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    let vocab_size = tokenizer.get_vocab_size(true);
    let txt = read_txt("assets/sub_wiki_0_99.txt");
    let seq_len = 32;
    let stride = 32;
    let batch_size = 2;
    let token_dataset = TokenDataset::new(txt, &tokenizer, 32, stride, &device)?;
    let mut dataloader = DataLoader::new(token_dataset, batch_size, true)?;
    let _ = dataloader.reset()?;
    let (x, y) = dataloader.next().unwrap()?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let embedding_dim = 32;
    let embedding = embedding(vocab_size, embedding_dim, vb.pp("embedding"))?;
    let encode = embedding.forward(&x)?;
    let rms_norm1 = RMSNorm::new(vb.pp("rms_norm1"), 1e-6, embedding_dim)?;
    let encode_rms = rms_norm1.forward(&encode)?;
    // let pos_embedding = SinusoidalPositonEmbedding::new(seq_len, embedding_dim, &device)?.pos_embedding;
    // let encode = encode.broadcast_add(&pos_embedding)?;
    // let attention1 = DotProductAttention::new(vb.pp("attention1"), embedding_dim, embedding_dim, &device)?;
    // // let output = attention1.forward(&encode)?;
    // let output = attention1.forward_with_mask(&encode, true)?;
    let n_head = 4;
    let n_kv_head = 2;
    let mut buffers = SharedBuffer::new()?;
    // let mha = MultiHeadAttention::new(vb.pp("mha1"), embedding_dim, embedding_dim, n_head, &device)?;
    // let output = mha.forward(&encode, true)?;
    // let output = mha.forward_with_rope(&encode, true)?;
    // let output = mha.forward_with_buffer(&encode, &mut buffers)?;
    let mut attention = GroupAttentionWithKVCache::new(
        vb.pp("self_attn"),
        embedding_dim,
        embedding_dim,
        n_head,
        n_kv_head,
        seq_len,
        &device,
    )?;
    let output = attention.forward(&encode_rms, &mut buffers, false, 0)?;
    println!("output {:?}", output);
    let res_cat = output.add(&encode)?;
    let rms_norm2 = RMSNorm::new(vb.pp("rms_norm2"), 1e-6, embedding_dim)?;
    let res_norm = rms_norm2.forward(&res_cat)?;
    let hidden_dim = 512;
    let feed_forward = FeedForward::new(
        vb.pp("feed_forward"),
        embedding_dim,
        hidden_dim,
        embedding_dim,
    )?;
    let feed_output = feed_forward.forward(&res_norm)?;
    println!("feed_output {:?}", feed_output);
    // let encode = tokenizer.encode("你好，你好可爱啊", true).map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    // let token_ids = encode.get_ids();
    // println!("{:?}", token_ids);
    // let token_tensor = Tensor::new(token_ids, &device)?;
    // let token_tensor = token_tensor.unsqueeze(0)?;
    // let encode = embedding.forward(&token_tensor)?;
    // let output = attention.forward(&encode, &mut buffers, true, 0)?;
    // println!("output: {:?}", output);
    // let pos_idx = token_ids.len();
    // let token_tensor = Tensor::new(3589 as u32, &device)?;
    // let token_tensor = token_tensor.unsqueeze(0)?.unsqueeze(0)?;  //(1, 1)
    // let encode = embedding.forward(&token_tensor)?;
    // let output = attention.forward(&encode, &mut buffers, true, pos_idx)?;
    // println!("output: {:?}", output);
    Ok(())
}

#[allow(unused)]
pub struct TokenDataset {
    input_ids: Tensor,
    target_ids: Tensor,
}

#[allow(unused)]
impl TokenDataset {
    pub fn new(
        txt: String,
        tokenizer: &Tokenizer,
        seq_len: usize,
        stride: usize,
        device: &Device,
    ) -> Result<Self> {
        let encoded = tokenizer
            .encode(txt, true)
            .map_err(|e| Error::Msg(format!("tokenizer encode error: {}", e)))?;
        let token_ids = encoded.get_ids();
        let token_len = token_ids.len();
        let max_token_id = token_len - seq_len;
        let mut input_ids_vec = Vec::new();
        let mut target_ids_vec = Vec::new();
        for i in (0..max_token_id).step_by(stride) {
            input_ids_vec.extend_from_slice(&token_ids[i..i + seq_len]);
            target_ids_vec.extend_from_slice(&token_ids[i + 1..i + seq_len + 1]);
        }
        let bs = input_ids_vec.len() / seq_len;
        let input_ids = Tensor::from_vec(input_ids_vec, (bs, seq_len), device)?;
        let target_ids = Tensor::from_vec(target_ids_vec, (bs, seq_len), device)?;
        Ok(Self {
            input_ids,
            target_ids,
        })
    }
}

#[allow(unused)]
pub fn apply_sin_cos(x: &Tensor, sin: &Tensor, cos: &Tensor) -> Result<Tensor> {
    let (_, _, _, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    let x2 = x2.affine(-1.0, 0.0)?;
    let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?;
    let x_cos = x.broadcast_mul(&cos)?;
    let x_sin = rotate_x.broadcast_mul(&sin)?;
    let rotate = x_cos.add(&x_sin)?;
    Ok(rotate)
}

#[allow(unused)]
pub struct RoPE {
    sin: Tensor,
    cos: Tensor,
}

#[allow(unused)]
impl RoPE {
    pub fn new(seq_len: usize, embedding_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(embedding_dim % 2, 0, "hidden_dim must be even");
        let pos_vec = (0..seq_len).map(|i| i as f32).collect();
        let angle_base_vec = (0..embedding_dim)
            .step_by(2)
            .map(|i| 1.0 as f32 / 10000.0_f32.powf(i as f32 / embedding_dim as f32))
            .collect();
        let pos = Tensor::from_vec(pos_vec, (seq_len, 1), device)?;
        let angle_base = Tensor::from_vec(angle_base_vec, (1, embedding_dim / 2), device)?;
        let angle_tensor = pos.matmul(&angle_base)?; //  (seq_len, embedding_dim/2)
        let angle_tensor = Tensor::cat(&[&angle_tensor, &angle_tensor], 1)?;
        let sin = angle_tensor.sin()?;
        let cos = angle_tensor.cos()?;
        Ok(Self { sin, cos })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rotate = apply_sin_cos(x, &self.sin, &self.cos)?;
        Ok(rotate)
        // let x_cos = x.broadcast_mul(&self.cos)?;
        // let dims = x.dims();
        // let half_dim = dims[dims.len()-1] / 2;
        // let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        // let x2  = x.narrow(D::Minus1, half_dim, half_dim)?;
        // let x2 = x2.affine(-1.0, 0.0)?;
        // let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?;
        // let x_sin = rotate_x.broadcast_mul(&self.sin)?;
        // let rotate = x_cos.add(&x_sin)?;
        // Ok(rotate)
    }

    pub fn apply(&self, x: &Tensor, pos_idx: usize) -> Result<Tensor> {
        // x != seq_len
        let (_, _, seq_len, _) = x.dims4()?;
        let (rope_seq_len, _) = self.cos.dims2()?;
        assert!(
            rope_seq_len >= (pos_idx + seq_len),
            " rope seq_le less than pos_id+seq_len"
        );
        let cos = self.cos.narrow(0, pos_idx, seq_len)?;
        let sin = self.sin.narrow(0, pos_idx, seq_len)?;
        let rotate = apply_sin_cos(x, &sin, &cos)?;
        Ok(rotate)
        // let x_cos = x.broadcast_mul(&cos)?;
        // let half_dim = head_dim / 2;
        // let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        // let x2  = x.narrow(D::Minus1, half_dim, half_dim)?;
        // let x2 = x2.affine(-1.0, 0.0)?;
        // let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?;
        // let x_sin = rotate_x.broadcast_mul(&sin)?;
        // let rotate = x_cos.add(&x_sin)?;
        // Ok(rotate)
    }

    pub fn new_roformer(seq_len: usize, embedding_dim: usize, device: &Device) -> Result<Self> {
        assert_eq!(embedding_dim % 2, 0, "hidden_dim must be even");
        let mut angle = Vec::new();
        for pos in 0..seq_len {
            for i in (0..embedding_dim).step_by(2) {
                let pos_i = pos as f32 / 10000.0_f32.powf(i as f32 / embedding_dim as f32);
                angle.extend_from_slice(&[pos_i, pos_i]);
            }
        }
        let angle_tensor = Tensor::from_vec(angle, (seq_len, embedding_dim), device)?;
        let cos = angle_tensor.cos()?;
        let sin = angle_tensor.sin()?;
        Ok(Self { sin, cos })
    }

    pub fn forward_roformer(&self, x: &Tensor) -> Result<Tensor> {
        let x_cos = x.broadcast_mul(&self.cos)?;
        let dims = x.dims();
        let mut new_dim = dims.to_vec();
        new_dim[dims.len() - 1] = dims[dims.len() - 1] / 2;
        new_dim.push(2);
        let x_reshape = x.reshape(new_dim)?; // x: (bs, seq_len, embedding_dim) -》 reshape: (bs, seq_len, embedding_dim/2, 2)
        let x1 = x_reshape.narrow(D::Minus1, 0, 1)?; // (bs, seq_len, embedding_dim/2, 1)
        let x2 = x_reshape.narrow(D::Minus1, 1, 1)?; // (bs, seq_len, embedding_dim/2, 1)
        let x2 = x2.affine(-1.0, 0.0)?;
        let rotate_stack_x = Tensor::stack(&[&x2, &x1], D::Minus1)?; // (bs, seq_len, embedding_dim/2, 1, 2)
        let rotate_flatten = rotate_stack_x.flatten(D::Minus(3), D::Minus1)?; // (bs, seq_len, embedding_dim)
        let x_sin = rotate_flatten.broadcast_mul(&self.sin)?;
        let rotate = x_cos.add(&x_sin)?;
        Ok(rotate)
    }
}

#[allow(unused)]
pub fn mask_filled(on_true: &Tensor, mask: &Tensor, on_false: f32) -> Result<Tensor> {
    let (mask_seq_len, _) = mask.dims2()?;
    let (_, _, seq_len, _) = on_true.dims4()?;
    assert!(
        mask_seq_len >= seq_len,
        "mask seq_len less than input data seq_len"
    );
    let mask = mask.i((..seq_len, ..seq_len))?;
    let mask = mask.broadcast_as(on_true.shape())?;
    let on_false = Tensor::new(on_false, on_true.device())?.broadcast_as(on_true.shape())?;
    let filled = mask.where_cond(on_true, &on_false)?;
    Ok(filled)
}

// key: seq_len, dim
// (Tensor, Rope)
#[allow(unused)]
pub struct SharedBuffer {
    buffers: HashMap<String, (Tensor, RoPE)>,
}

#[allow(unused)]
impl SharedBuffer {
    pub fn new() -> Result<Self> {
        let buffers: HashMap<String, (Tensor, RoPE)> = HashMap::new();
        Ok(Self { buffers })
    }
    pub fn get(&mut self, seq_len: usize, dim: usize, device: &Device) -> Result<&(Tensor, RoPE)> {
        let key = format!("{}_{}", seq_len, dim);
        if !self.buffers.contains_key(&key) {
            let mask = Tensor::tril2(seq_len, candle_core::DType::U32, device)?;
            let rope = RoPE::new(seq_len, dim, device)?;
            self.buffers.insert(key.clone(), (mask, rope));
        }
        let value = self
            .buffers
            .get(&key)
            .ok_or(Error::Msg(format!("get mask rope key:{} None", key)))?;
        Ok(value)
    }
}

// kv cache --推理时才需要
// use_cache: bool
// cache_k, cache_v
// pos_id
#[allow(unused)]
pub struct GroupAttentionWithKVCache {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    out_proj: Linear,
    n_head: usize,
    n_kv_head: usize,
    group_size: usize,
    head_dim: usize,
    out_dim: usize,
    max_context: usize,
    d_sqrt: Tensor,
    cache_k: Option<Tensor>,
    cache_v: Option<Tensor>,
}

#[allow(unused)]
impl GroupAttentionWithKVCache {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        max_context: usize,
        device: &Device,
    ) -> Result<Self> {
        assert_eq!(out_dim % n_head, 0, "out_dim must be divisible by n_head");
        assert_eq!(
            n_head % n_kv_head,
            0,
            "n_head must be divisible by n_kv_head"
        );
        let head_dim = out_dim / n_head;
        let w_q = linear_no_bias(in_dim, out_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(in_dim, n_kv_head * head_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(in_dim, n_kv_head * head_dim, vb.pp("w_v"))?;
        let out_proj = linear_no_bias(out_dim, out_dim, vb.pp("out_proj"))?;
        let group_size = n_head / n_kv_head;
        let d_sqrt = 1.0 / (head_dim as f32).sqrt();
        let d_sqrt = Tensor::new(d_sqrt, device)?;
        Ok(Self {
            w_q,
            w_k,
            w_v,
            out_proj,
            n_head,
            n_kv_head,
            group_size,
            head_dim,
            out_dim,
            max_context,
            d_sqrt,
            cache_k: None,
            cache_v: None,
        })
    }
    pub fn forward(
        &mut self,
        x: &Tensor,
        buffers: &mut SharedBuffer,
        use_cache: bool,
        pos_idx: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;
        let (mask, rope) = buffers.get(self.max_context, self.head_dim, x.device())?;
        // (bs, n_head, seq_len, head_dim)
        let mut q = self
            .w_q
            .forward(x)?
            .reshape((bs, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = self
            .w_k
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = self
            .w_v
            .forward(x)?
            .reshape((bs, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        if use_cache {
            q = rope.apply(&q, pos_idx)?;
            k = rope.apply(&k, pos_idx)?;
            if self.cache_k.is_none() {
                self.cache_k = Some(k.clone());
                self.cache_v = Some(v.clone());
            } else {
                if let Some(cache_k_) = &self.cache_k {
                    k = Tensor::cat(&[cache_k_, &k], D::Minus2)?;
                    self.cache_k = Some(k.clone());
                }
                if let Some(cache_v_) = &self.cache_v {
                    v = Tensor::cat(&[cache_v_, &v], D::Minus2)?;
                    self.cache_v = Some(v.clone());
                }
            }
        } else {
            q = rope.forward(&q)?;
            k = rope.forward(&k)?;
        }
        let k = k.repeat((1, self.group_size, 1, 1))?;
        let v = v.repeat((1, self.group_size, 1, 1))?;
        let mut atten_score = q.matmul(&k.t()?)?;
        if seq_len != 1 {
            atten_score = mask_filled(&atten_score, &mask, f32::NEG_INFINITY)?;
        }
        let atten_score = atten_score.broadcast_mul(&self.d_sqrt)?;
        let softmax = ops::softmax(&atten_score, D::Minus1)?;
        let atten_weight = softmax.matmul(&v)?; // (bs, n_head, seq_len, head_dim)
        let atten_weight = atten_weight
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.out_dim))?;
        let atten_weight = self.out_proj.forward(&atten_weight)?;
        Ok(atten_weight)
    }

    pub fn reset_kv_cache(&mut self) {
        self.cache_k = None;
        self.cache_v = None;
    }
}

#[allow(unused)]
pub struct RMSNorm {
    weight: Tensor,
    eps: Tensor,
}

#[allow(unused)]
impl RMSNorm {
    pub fn new(vb: VarBuilder, eps: f32, dim: usize) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", Init::Const(1.0))?;
        let eps = Tensor::new(eps, vb.device())?;
        Ok(Self { weight, eps })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.powf(2.0)?.mean(D::Minus1)?; // 2,32, 32-> 2,32, 1
        let rms = mean
            .broadcast_add(&self.eps)?
            .sqrt()?
            .unsqueeze(D::Minus1)?;
        let x_norm = x.broadcast_div(&rms)?;
        let x_norm = x_norm.broadcast_mul(&self.weight)?;
        Ok(x_norm)
    }
}

#[allow(unused)]
pub struct FeedForward {
    up: Linear,
    gate: Linear,
    down: Linear,
}

#[allow(unused)]
impl FeedForward {
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        let up = linear_no_bias(in_dim, hidden_dim, vb.pp("up"))?;
        let gate = linear_no_bias(in_dim, hidden_dim, vb.pp("gate"))?;
        let down = linear_no_bias(hidden_dim, out_dim, vb.pp("down"))?;
        Ok(Self { up, gate, down })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up_x = self.up.forward(x)?;
        let gate_x = self.gate.forward(x)?.silu()?;
        let mul_cat = up_x.mul(&gate_x)?;
        let down = self.down.forward(&mul_cat)?;
        Ok(down)
    }
}

#[allow(unused)]
pub struct LLMConfig {
    vocab_size: usize,
    embedding_dim: usize,
    n_block: usize,
    n_head: usize,
    n_kv_head: usize,
    max_context: usize,
    hidden_dim: usize,
    eps: f32,
}

#[allow(unused)]
impl LLMConfig {
    pub fn default() -> Result<Self> {
        Ok(Self {
            vocab_size: 151669,
            embedding_dim: 512,
            n_block: 8,
            n_head: 8,
            n_kv_head: 4,
            max_context: 256,
            hidden_dim: 1024,
            eps: 1e-6,
        })
    }
}

#[allow(unused)]
pub struct AttentionBlock {
    rms_norm1: RMSNorm,
    attention: GroupAttentionWithKVCache,
    rms_norm2: RMSNorm,
    feed_forward: FeedForward,
}

#[allow(unused)]
impl AttentionBlock {
    pub fn new(vb: VarBuilder, config: &LLMConfig) -> Result<Self> {
        let rms_norm1 = RMSNorm::new(vb.pp("rms_norm1"), config.eps, config.embedding_dim)?;
        let attention = GroupAttentionWithKVCache::new(
            vb.pp("attention"),
            config.embedding_dim,
            config.embedding_dim,
            config.n_head,
            config.n_kv_head,
            config.max_context,
            vb.device(),
        )?;
        let rms_norm2 = RMSNorm::new(vb.pp("rms_norm2"), config.eps, config.embedding_dim)?;
        let feed_forward = FeedForward::new(
            vb.pp("feed_forward"),
            config.embedding_dim,
            config.hidden_dim,
            config.embedding_dim,
        )?;
        Ok(Self {
            rms_norm1,
            attention,
            rms_norm2,
            feed_forward,
        })
    }
    pub fn forward(
        &mut self,
        x: &Tensor,
        buffers: &mut SharedBuffer,
        use_cache: bool,
        pos_idx: usize,
    ) -> Result<Tensor> {
        let x_norm1 = self.rms_norm1.forward(x)?;
        let x_atten = self
            .attention
            .forward(&x_norm1, buffers, use_cache, pos_idx)?;
        let shortcut = x.add(&x_atten)?;
        let x_norm2 = self.rms_norm2.forward(&shortcut)?;
        let x_feed = self.feed_forward.forward(&x_norm2)?;
        let shortcut = shortcut.add(&x_feed)?;
        Ok(shortcut)
    }
    pub fn reset_kv_cache(&mut self) {
        self.attention.reset_kv_cache();
    }
}

#[allow(unused)]
pub struct LLM {
    embedding: Embedding,
    attention_blocks: Vec<AttentionBlock>,
    final_rms: RMSNorm,
    out_proj: Linear,
    buffers: SharedBuffer,
}

#[allow(unused)]
impl LLM {
    pub fn new(vb: VarBuilder, config: &LLMConfig) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.embedding_dim, vb.pp("embedding"))?;
        let mut attention_blocks = Vec::new();
        for i in 0..config.n_block {
            let block = AttentionBlock::new(vb.pp(format!("block_{}", i)), config)?;
            attention_blocks.push(block);
        }
        let final_rms = RMSNorm::new(vb.pp("final_rms"), config.eps, config.embedding_dim)?;
        let out_proj = linear_no_bias(config.embedding_dim, config.vocab_size, vb.pp("out_proj"))?;
        let buffers = SharedBuffer::new()?;
        Ok(Self {
            embedding,
            attention_blocks,
            final_rms,
            out_proj,
            buffers,
        })
    }
    pub fn forward(&mut self, x: &Tensor, use_cache: bool, pos_idx: usize) -> Result<Tensor> {
        let mut x = self.embedding.forward(x)?;
        for block in &mut self.attention_blocks {
            x = block.forward(&x, &mut self.buffers, use_cache, pos_idx)?;
        }
        let x = self.final_rms.forward(&x)?;
        let x = self.out_proj.forward(&x)?;
        Ok(x)
    }

    pub fn reset_kv_cache(&mut self) {
        for block in &mut self.attention_blocks {
            block.reset_kv_cache();
        }
    }
}

#[allow(unused)]
pub fn encode_str(str: &str, tokenizers: &Tokenizer, device: &Device) -> Result<Tensor> {
    let encode = tokenizers
        .encode(str, true)
        .map_err(|e| Error::Msg(format!("tokenizer encode error{}", e)))?;
    let token_ids = encode.get_ids();
    let len = token_ids.len();
    let tensor = Tensor::from_slice(token_ids, (1, len), device)?;
    Ok(tensor)
}

#[allow(unused)]
pub fn decode_tokens(token_ids: &Tensor, tokenizers: &Tokenizer) -> Result<String> {
    let token_ids_vec = match token_ids.rank() {
        1 => token_ids.to_vec1()?,
        2 => token_ids.squeeze(0)?.to_vec1()?,
        _ => {
            return Err(Error::Msg(format!(
                "can't active this rank {} Tensor",
                token_ids.rank()
            )));
        }
    };
    let decode = tokenizers
        .decode(&token_ids_vec, true)
        .map_err(|e| Error::Msg(format!("tokenizer encode error{}", e)))?;
    Ok(decode)
}

#[allow(unused)]
pub fn generate_simple(
    model: &mut LLM,
    idx: &Tensor,
    max_generate: usize,
    max_context: usize,
) -> Result<Tensor> {
    model.reset_kv_cache();
    let mut idx = idx.clone();
    let (_, num_tokens) = idx.dims2()?;
    if num_tokens > max_context {
        let start = num_tokens - max_context;
        idx = idx.i((.., start..num_tokens))?;
    }
    let mut pos_idx = 0;
    let mut logits = model.forward(&idx, true, pos_idx)?;
    for _ in 0..max_generate {
        let (_, n_token, _) = logits.dims3()?;
        pos_idx += n_token;
        if pos_idx > max_context {
            pos_idx = 0;
        }
        logits = logits.i((.., n_token - 1, ..))?;
        let probs = ops::softmax(&logits, D::Minus1)?;
        let mut idx_next = probs.argmax(D::Minus1)?;
        if idx_next.rank() == 1 {
            idx_next = idx_next.unsqueeze(0)?;
        }
        idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?;
        logits = model.forward(&idx_next, true, pos_idx)?;
    }
    model.reset_kv_cache();
    Ok(idx)
}

#[allow(unused)]
pub fn generate_print_txt(
    model: &mut LLM,
    tokenizers: &Tokenizer,
    start_context: &str,
    max_generate: usize,
    max_context: usize,
    device: &Device,
) -> Result<()> {
    let encode = encode_str(start_context, tokenizers, device)?;
    let generate_tokens = generate_simple(model, &encode, max_generate, max_context)?;
    let str = decode_tokens(&generate_tokens, tokenizers)?;
    println!("generate: \n{:?}", str);
    Ok(())
}

// batch loss backward
// epoch , train loader -> batch loss backward
// global step  val_loader loss | generate->
// model, optimizer, train_loader, val_loader,tokeizers, start_context, max_generate, max_context, device

#[allow(unused)]
pub fn get_batch_loss(model: &mut LLM, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let logits = model.forward(x, false, 0)?;
    let loss_ = loss::cross_entropy(&logits.flatten(0, 1)?, &y.flatten_all()?)?;
    Ok(loss_)
}

#[allow(unused)]
pub fn get_loader_loss(model: &mut LLM, dataloader: &mut DataLoader) -> Result<f32> {
    let mut loss_sum = 0.0;
    let mut count = 0;
    for batch in dataloader {
        let (x, y) = batch?;
        let loss_ = get_batch_loss(model, &x, &y)?;
        loss_sum += loss_.to_scalar::<f32>()?;
        count += 1;
    }
    let loss_ = loss_sum / count as f32;
    Ok(loss_)
}

#[allow(unused)]
pub fn train_model(
    model: &mut LLM,
    train_loader: &mut DataLoader,
    val_loader: &mut DataLoader,
    optimizer: &mut AdamW,
    tokenizers: &Tokenizer,
    epochs: usize,
    eval_step: usize,
    start_context: &str,
    max_generate: usize,
    max_context: usize,
    device: &Device,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut train_loss_vec = Vec::new();
    let mut val_loss_vec = Vec::new();
    let mut global_step = 0;
    for epoch in 0..epochs {
        let _ = train_loader.reset()?;
        for batch in &mut *train_loader {
            let (x, y) = batch?;
            let loss_ = get_batch_loss(model, &x, &y)?;
            let _ = optimizer.backward_step(&loss_)?;
            global_step += 1;
            if global_step % eval_step == 0 {
                let _ = val_loader.reset()?;
                let val_loss = get_loader_loss(model, val_loader)?;
                let train_loss = loss_.to_scalar::<f32>()?;
                println!(
                    "global_step: {} train_loss: {}, val_loss: {}",
                    global_step, train_loss, val_loss
                );
                train_loss_vec.push(train_loss);
                val_loss_vec.push(val_loss);
                let _ = generate_print_txt(
                    model,
                    tokenizers,
                    start_context,
                    max_generate,
                    max_context,
                    device,
                )?;
            }
        }
        let _ = train_loader.reset()?;
        let _ = val_loader.reset()?;
        let val_loss = get_loader_loss(model, val_loader)?;
        let train_loss = get_loader_loss(model, train_loader)?;
        println!(
            "epoch: {} train_loss: {}, val_loss: {}",
            epoch, train_loss, val_loss
        );
        train_loss_vec.push(train_loss);
        val_loss_vec.push(val_loss);
    }
    Ok((train_loss_vec, val_loss_vec))
}

#[allow(unused)]
pub fn train_main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let tokenizer = Tokenizer::from_file("assets/tokenizer.json")
        .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    let batch_size = 2;
    let config = LLMConfig::default()?;

    let val_txt = read_txt("assets/sub_wiki_0_99.txt");
    let val_dataset = TokenDataset::new(
        val_txt,
        &tokenizer,
        config.max_context,
        config.max_context,
        &device,
    )?;
    let mut val_loader = DataLoader::new(val_dataset, batch_size, true)?;

    let train_txt = read_txt("assets/sub_wiki_0_99.txt");
    let train_dataset = TokenDataset::new(
        train_txt,
        &tokenizer,
        config.max_context,
        config.max_context,
        &device,
    )?;
    let mut train_loader = DataLoader::new(train_dataset, batch_size, true)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let mut llm = LLM::new(vb, &config)?;
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), 1e-4)?;
    let (train_loss, val_loss) = train_model(
        &mut llm,
        &mut train_loader,
        &mut val_loader,
        &mut optimizer,
        &tokenizer,
        2,
        10,
        "绿色建筑或绿建筑（green building），是指本身及其使用过程在生命周期中",
        100,
        config.max_context,
        &device,
    )?;
    let _ = plot_loss_curves(&train_loss, &val_loss, "loss_curves.png", "loss curve")?;
    Ok(())
}

#[allow(unused)]
pub fn qwen_generate_print(
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    logits_processor: &mut LogitsProcessor,
    prompt: &str,
    sample_len: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    device: &Device,
) -> Result<()> {
    let mut tokens = encode_str(prompt, tokenizer, device)?
        .squeeze(0)?
        .to_vec1()?;
    let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>").copied() {
        Some(token) => token,
        None => return Err(Error::Msg("can't get <|endoftext|>".to_string())),
    };
    let eos_token2 = match tokenizer.get_vocab(true).get("<|im_end|>").copied() {
        Some(token) => token,
        None => return Err(Error::Msg("can't get <|im_end|>".to_string())),
    };
    let mut generated_tokens = 0usize;
    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };
        // 【0.1， 0.2， 0.1， 0.1， 0.3， 0.2】
        //  【0.1， 0.3，0.4， 0.5， 0.8， 1.0】
        // rand-> 0.53, 0.05

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;
        if next_token == eos_token || next_token == eos_token2 {
            break;
        }
        let decode = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| Error::Msg(format!("tokenizer encode error{}", e)))?;

        print!("{decode}");
        std::io::stdout().flush()?;
    }
    println!("\n{generated_tokens} tokens generated");
    model.clear_kv_cache();
    Ok(())
}

#[allow(unused)]
pub fn test_qwen3() -> Result<()> {
    let tokenizer_file = "/mnt/c/jhq/huggingface_model/Qwen/Qwen3-0___6B/tokenizer.json";
    let weight_file = "/mnt/c/jhq/huggingface_model/Qwen/Qwen3-0___6B/model.safetensors";
    let config_file = "/mnt/c/jhq/huggingface_model/Qwen/Qwen3-0___6B/config.json";
    let tokenizer = Tokenizer::from_file(tokenizer_file)
        .map_err(|e| Error::Msg(format!("tokenizer from file error:{}", e)))?;
    let device = Device::cuda_if_available(0)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weight_file], candle_core::DType::F16, &device)?
    };
    let config: Config =
        serde_json::from_slice(&std::fs::read(config_file)?).expect("config file read failed");
    let mut model = ModelForCausalLM::new(&config, vb)?;

    let mut logits_processor = LogitsProcessor::new(299792458, None, None);
    loop {
        println!("我是智能小助手，有什么能帮到你的吗？输入quit退出");
        let mut start_context = String::new();
        std::io::stdin()
            .read_line(&mut start_context)
            .expect("无法读取输入");
        if start_context.trim().eq("quit") {
            break;
        }
        start_context += "<think></think>";
        let _ = qwen_generate_print(
            &mut model,
            &tokenizer,
            &mut logits_processor,
            &start_context,
            10000,
            1.1,
            64,
            &device,
        )?;
    }

    Ok(())
}
