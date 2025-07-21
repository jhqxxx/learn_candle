use std::{
    fs, path::Path, sync::{Arc, Mutex}
};

use candle_core::{D, DType, Device, Error, Module, Result, Tensor};
use candle_nn::{
    AdamW, Conv2d, Conv2dConfig, Dropout, Embedding, Init, LayerNorm, Linear, Optimizer,
    VarBuilder, VarMap, conv2d, embedding, layer_norm, linear, loss, ops,
};
use image::{
    DynamicImage, GenericImage, GenericImageView, ImageBuffer, ImageReader, Rgb, RgbImage, Rgba,
    imageops::{self, FilterType},
};
use rand::{Rng, random_bool};
use rayon::prelude::*;
use crate::utils::dataset::{DataLoader, ImgDataset};
use crate::utils::net::{CustomModule, plot_loss_curves};
use walkdir::WalkDir;
use candle_einops::einops;
use crate::chapter3::MultiHeadAttention;
use std::f32::consts::PI;

/// 随机裁剪
#[allow(unused)]
pub fn random_crop(img: &DynamicImage, width: u32, height: u32) -> Option<DynamicImage> {
    let short = img.width().min(img.height());
    let mut rng = rand::rng();
    let min_len = short / 2;
    let sub_len = rng.random_range(min_len..short);
    let x = rng.random_range(0..img.width() - sub_len);
    let y = rng.random_range(0..img.height() - sub_len);
    match img {
        DynamicImage::ImageRgb8(rgb) => {
            let sub_image = imageops::crop_imm(rgb, x, y, sub_len, sub_len).to_image();
            let sub_image = DynamicImage::ImageRgb8(sub_image).resize_exact(
                width,
                height,
                FilterType::CatmullRom,
            );
            return Some(sub_image);
        }
        DynamicImage::ImageRgba8(rgba) => {
            let sub_image = imageops::crop_imm(rgba, x, y, sub_len, sub_len).to_image();
            let sub_image = DynamicImage::ImageRgba8(sub_image).resize_exact(
                width,
                height,
                FilterType::CatmullRom,
            );
            return Some(sub_image);
        }
        _ => {
            println!("crop image not is rgb8");
            return None;
        }
    }
}

/// 随机旋转（-30° 到 +30°）
#[allow(unused)]
pub fn random_rotation(img: &DynamicImage) -> DynamicImage {
    let angle_deg = rand::rng().random_range(-30.0..30.0);
    let angle_rad = angle_deg as f32 * PI / 180.0;

    // 创建空白画布
    let (w, h) = (img.width() as f32, img.height() as f32);
    let new_size = (
        (w * angle_rad.abs().cos() + h * angle_rad.abs().sin()) as u32,
        (h * angle_rad.abs().cos() + w * angle_rad.abs().sin()) as u32,
    );

    let canvas: ImageBuffer<Rgb<u8>, Vec<u8>> =
        RgbImage::from_pixel(new_size.0, new_size.1, Rgb([255u8, 255u8, 255u8]));
    let mut canvas = DynamicImage::ImageRgb8(canvas);
    // 计算旋转中心
    let center_x = w / 2.0;
    let center_y = h / 2.0;
    let new_center_x = new_size.0 as f32 / 2.0;
    let new_center_y = new_size.1 as f32 / 2.0;

    // 旋转
    for y in 0..new_size.1 {
        for x in 0..new_size.0 {
            let dx = x as f32 - new_center_x;
            let dy = y as f32 - new_center_y;

            let src_x = (dx * angle_rad.cos() - dy * angle_rad.sin() + center_x) as i32;
            let src_y = (dx * angle_rad.sin() + dy * angle_rad.cos() + center_y) as i32;

            if src_x >= 0 && src_x < w as i32 && src_y >= 0 && src_y < h as i32 {
                let pixel = img.get_pixel(src_x as u32, src_y as u32);
                canvas.put_pixel(x, y, pixel);
            }
        }
    }
    canvas = canvas.resize(img.width(), img.height(), FilterType::CatmullRom);
    canvas
}

/// 随机擦除
#[allow(unused)]
pub fn random_erase(img: &DynamicImage) -> DynamicImage {
    let mut erase_img = img.clone();
    let mut rng = rand::rng();
    let min_erase_size = 10;
    let max_erase_ration = 0.25;
    let erase_w = rng.random_range(min_erase_size..=(img.width() as f32 * max_erase_ration) as u32);
    let erase_h =
        rng.random_range(min_erase_size..=(img.height() as f32 * max_erase_ration) as u32);
    let erase_start_x = rng.random_range(0..(img.width() - erase_w));
    let erase_start_y = rng.random_range(0..img.height() - erase_h);
    for x in erase_start_x..erase_start_x + erase_w {
        for y in erase_start_y..erase_start_y + erase_h {
            let r = rng.random_range(0..255) as u8;
            let g = rng.random_range(0..255) as u8;
            let b = rng.random_range(0..255) as u8;
            erase_img.put_pixel(x, y, Rgba([r, g, b, 255u8]));
        }
    }
    erase_img
}

/// resize图像，保持图像原比例，填充全0像素成目标宽高
#[allow(unused)]
pub fn resize_with_edge_padding(img: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    // 按图像原比例resize,可能不是输入的宽高
    let mut img = img.resize(width, height, FilterType::CatmullRom);
    // 使用全0像素填充为输入宽高
    if img.height() != height || img.width() != width {
        let (img_h, img_w) = (img.height(), img.width());
        let img_buffer = img.to_rgb8();
        let mut canvas: ImageBuffer<Rgb<u8>, Vec<u8>> =
            RgbImage::from_pixel(width, height, Rgb([255u8, 255u8, 255u8]));
        let x_offset = (width - img_w) / 2;
        let y_offset = (height - img_h) / 2;
        imageops::overlay(&mut canvas, &img_buffer, x_offset as i64, y_offset as i64);
        img = DynamicImage::ImageRgb8(canvas);
    }
    img
}

#[allow(unused)]
pub fn get_img_vec(path: &Path, data_type: &str) -> Result<Vec<Vec<u8>>> {
    let mut img = ImageReader::open(path)
        .map_err(|e| Error::Msg(format!("open file error:{}", e)))?
        .decode()
        .map_err(|e| Error::Msg(format!("decode img error:{}", e)))?;
    let mut img_vec = Vec::new();
    if data_type == "train" {
        if random_bool(1.0 / 4.0) {
            // 随机裁剪
            if let Some(crop_img) = random_crop(&img, 224, 224) {
                img_vec.push(crop_img.to_rgb8().into_raw());
            }
            // println!(" crop img height{}, width:{}", img.height(), img.width());
        }
        img = resize_with_edge_padding(&img, 224, 224);
        img_vec.push(img.clone().to_rgb8().into_raw());
        if random_bool(1.0 / 4.0) {
            //水平翻转
            let clip_img = img.fliph();
            img_vec.push(clip_img.to_rgb8().into_raw());
            // println!("flip img height{}, width:{}", img.height(), img.width());
        }
        if random_bool(1.0 / 4.0) {
            // 随机旋转
            let rotate_img = random_rotation(&img);
            img_vec.push(rotate_img.to_rgb8().into_raw());
            // println!("rotation img height{}, width:{}", img.height(), img.width());
        }
        if random_bool(1.0 / 4.0) {
            let erase_img = random_erase(&img);
            img_vec.push(erase_img.to_rgb8().into_raw());
        }
    } else {
        img = resize_with_edge_padding(&img, 224, 224);
        img_vec.push(img.to_rgb8().into_raw());
    }
    Ok(img_vec)
}

#[allow(unused)]
pub fn get_img_dataset(
    image_path: &str,
    data_group: Vec<&str>,
    class_num: usize,
    device: &Device,
    start: Option<usize>,
    end: Option<usize>,
) -> Result<Vec<ImgDataset>> {
    let mut data_dataset_vec = Vec::new();
    for group in data_group {
        let img_vec = Arc::new(Mutex::new(Vec::new()));
        let label_vec = Arc::new(Mutex::new(Vec::new()));
        for i in 0..class_num {
            let class_path = format!("{image_path}/{group}/{i}");
            let files: Vec<_> = WalkDir::new(class_path)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|file| file.file_type().is_file())
                .collect();

            let img_vec_clone = Arc::clone(&img_vec);
            let label_vec_clone = Arc::clone(&label_vec);
            files.into_par_iter().for_each(move |file| {
                if start.is_some() || end.is_some() {
                    let path_str = file.path().file_name().unwrap().to_str().unwrap();
                    let idx = path_str
                        .split(".")
                        .next()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    if start.is_some() && idx < start.unwrap() {
                        return;
                    }
                    if end.is_some() && idx >= end.unwrap() {
                        return;
                    }
                }
                match get_img_vec(file.path(), group) {
                    Ok(img_vec_u8) => {
                        let img_len = img_vec_u8.len();
                        img_vec_clone.lock().unwrap().extend_from_slice(&img_vec_u8);
                        let label_vec = vec![i as u8; img_len];
                        label_vec_clone
                            .lock()
                            .unwrap()
                            .extend_from_slice(&label_vec);
                    }
                    Err(e) => {
                        println!("file: {:?} error: {:?}", file.path(), e);
                    }
                }
            });
        }
        let img_vec = Arc::try_unwrap(img_vec).unwrap().into_inner().unwrap();
        let label_vec = Arc::try_unwrap(label_vec).unwrap().into_inner().unwrap();
        let img_dataset = ImgDataset::new(img_vec, label_vec, device)?;
        data_dataset_vec.push(img_dataset);
    }
    Ok(data_dataset_vec)
}

#[allow(unused)]
pub struct PatchConv {
    conv: Conv2d,
}

#[allow(unused)]
impl PatchConv {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 0,
            stride,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };

        // (bs, c, h, w) -> conv
        // img  224*224
        // patch - 16*16  3 -> 768
        // patch_num: 224/16 = 14, 14*14=196
        // (bs, c, h, w) -> (bs, seq_len, embedding_dim) -> seq_len =196, embeding_dim=768
        // ksize=16, stride=16
        // (224 - 16 + 2*padding) / stride + 1  => 14
        // (bs, 768, 14, 14)
        let conv = conv2d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (bs, 3, 224, 224) -> (bs, out_channel, 14, 14)
        let x = self.conv.forward(x)?;
        let x = x.flatten(2, 3)?; // (bs, out_channel, 196)
        let x = x.transpose(1, 2)?; // (bs, 196, out_channel)
        Ok(x)
    }
}

#[allow(unused)]
pub struct PatchLinear {
    linear: Linear,
}

#[allow(unused)]
impl PatchLinear {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let linear = linear(in_dim, out_dim, vb.pp("linear"))?;
        Ok(Self { linear })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (bs, 3, 224, 224) -> (bs, 196, 768)
        // let x = x.permute((0, 2, 3, 1))?.contiguous()?;  // (bs, 3, 224, 224) -> (bs, 224, 224, 3)        
        // python -> ("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = 16, p2 = 16)
        // let x = einops!("b (h p1:16) (w p2:16) c -> b (h w) (p1 p2 c)", &x); // (bs, 224, 224, 3) -> (bs, 196, 768)
        let x = einops!("b c (h p1:16) (w p2:16) -> b (h w) (p1 p2 c)", &x); // (bs, 3, 224, 224) -> (bs, 196, 768)
        let x = self.linear.forward(&x)?;
        Ok(x)
    }
}

#[allow(unused)]
pub struct MLP {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
}

#[allow(unused)]
impl MLP {
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        let linear1 = linear(in_dim, hidden_dim, vb.pp("linear1"))?;
        let linear2 = linear(hidden_dim, out_dim, vb.pp("linear2"))?;
        let dropout = Dropout::new(0.2);
        Ok(Self {
            linear1,
            linear2,
            dropout,
        })
    }
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.gelu()?;
        let x = self.dropout.forward(&x, train)?;
        let x = self.linear2.forward(&x)?;
        Ok(x)
    }
}

#[allow(unused)]
pub struct ViTBlock {
    attn: MultiHeadAttention,
    mlp: MLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn_dropout: Dropout,
}

#[allow(unused)]
impl ViTBlock {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        n_head: usize,
    ) -> Result<Self> {
        let attn = MultiHeadAttention::new(vb.pp("attn"), in_dim, out_dim, n_head, vb.device())?;
        let mlp = MLP::new(vb.pp("mlp"), in_dim, hidden_dim, out_dim)?;
        let norm1 = layer_norm(out_dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = layer_norm(out_dim, 1e-6, vb.pp("norm2"))?;
        let attn_dropout = Dropout::new(0.1);
        Ok(Self {
            attn,
            mlp,
            norm1,
            norm2,
            attn_dropout,
        })
    }
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let shortcut = x.clone();
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x, false)?;
        let x = self.attn_dropout.forward(&x, train)?;
        let shortcut = shortcut.add(&x)?;
        let x = self.norm2.forward(&shortcut)?;
        let x = self.mlp.forward(&x, train)?;
        let x = shortcut.add(&x)?;
        Ok(x)
    }
}

#[allow(unused)]
pub struct ViT {
    // patch_embed: PatchConv,
    patch_embed: PatchLinear,
    pos_embedding: Embedding,
    ln_pre: LayerNorm,
    blocks: Vec<ViTBlock>,
    ln_post: LayerNorm,
    head: MLP,
}

#[allow(unused)]
impl ViT {
    pub fn new(
        vb: VarBuilder,
        image_size: usize,
        patch_size: usize,
        hidden_dim: usize,
        n_block: usize,
        n_head: usize,
        n_class: usize,
    ) -> Result<Self> {
        assert!(
            image_size % patch_size == 0,
            "image size must be divisible by patch size"
        );
        let n_patch = image_size * image_size / patch_size / patch_size;
        let embed_dim = patch_size * patch_size * 3;
        // let patch_embed = PatchConv::new(vb.pp("patch"), 3, embed_dim, patch_size, patch_size)?;
        let patch_embed = PatchLinear::new(vb.pp("patch"), embed_dim, embed_dim)?;
        let pos_embedding = embedding(n_patch, embed_dim, vb.pp("pos_embedding"))?;
        let ln_pre = layer_norm(embed_dim, 1e-6, vb.pp("ln_pre"))?;
        let mut blocks = Vec::new();
        for i in 0..n_block {
            let block = ViTBlock::new(
                vb.pp(format!("block_{}", i)),
                embed_dim,
                hidden_dim,
                embed_dim,
                n_head,
            )?;
            blocks.push(block);
        }
        let ln_post = layer_norm(embed_dim, 1e-6, vb.pp("ln_pre"))?;
        // let head = linear(embed_dim, n_class, vb.pp("head"))?;
        let head = MLP::new(vb.pp("head"), embed_dim, embed_dim / 2, n_class)?;
        Ok(Self {
            patch_embed,
            pos_embedding,
            ln_pre,
            blocks,
            ln_post,
            head,
        })
    }
}

#[allow(unused)]
impl CustomModule for ViT {
    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let patch_embed = self.patch_embed.forward(x)?; // (bs, n_patch, embed_dim)
        let pos_embed = self.pos_embedding.embeddings().unsqueeze(0)?; //(1, n_patch, embed_dim)
        let mut x = patch_embed.broadcast_add(&pos_embed)?;
        x = self.ln_pre.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x, train)?;
        }
        x = self.ln_post.forward(&x)?;
        // (bs, n_patch, embed_dim) -> (bs, embed_dim)
        let x = x.mean(D::Minus2)?;
        // (bs, embed_dim) -> (bs, embed_dim / 2) -> (bs, n_cls)
        let x = self.head.forward(&x, train)?;
        Ok(x)
    }
}

#[allow(unused)]
pub struct CustomLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    data_format: String,
}

#[allow(unused)]
impl CustomLayerNorm {
    pub fn new(vb: VarBuilder, size: usize, eps: f64, data_format: &str) -> Result<Self> {
        if !["channels_last", "channels_first"].contains(&data_format) {
            return Err(Error::Msg(format!(
                "data_format type {} not surpose",
                data_format
            )));
        }
        let weight = vb.get_with_hints(size, "weight", Init::Const(1.))?;
        let bias = vb.get_with_hints(size, "bias", Init::Const(0.))?;
        Ok(Self {
            weight,
            bias,
            eps,
            data_format: data_format.to_string(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self.data_format.as_str() {
            "channels_last" => Ok(ops::layer_norm(
                x,
                &self.weight,
                &self.bias,
                self.eps as f32,
            )?),
            "channels_first" => {
                let mean = x.mean_keepdim(1)?;
                let standard = x
                    .broadcast_sub(&mean)?
                    .powf(2.0)?
                    .mean_keepdim(1)?
                    .affine(1.0, self.eps)?
                    .sqrt()?;
                let x = x.broadcast_sub(&mean)?.broadcast_div(&standard)?;
                let x = x
                    .broadcast_mul(&self.weight.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?
                    .broadcast_add(&self.bias.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?;
                Ok(x)
            }
            _ => Err(Error::Msg(format!(
                "data_format type {} not surpose",
                self.data_format
            ))),
        }
    }
}

#[allow(unused)]
pub struct Stem {
    conv: Conv2d,
    norm: CustomLayerNorm,
}

#[allow(unused)]
impl Stem {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 0,
            stride,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let conv = conv2d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        let norm = CustomLayerNorm::new(vb.pp("layer_norm"), out_channels, 1e-6, "channels_first")?;
        Ok(Self { conv, norm })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        Ok(x)
    }
}

#[allow(unused)]
pub struct DownSample {
    norm: CustomLayerNorm,
    conv: Conv2d,
}

#[allow(unused)]
impl DownSample {
    pub fn new(
        vb: VarBuilder,
        in_channel: usize,
        out_channel: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Result<Self> {
        let norm = CustomLayerNorm::new(vb.pp("norm"), in_channel, 1e-6, "channels_first")?;
        let cfg = Conv2dConfig {
            padding: 0,
            stride: stride,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let conv = conv2d(in_channel, out_channel, kernel_size, cfg, vb.pp("conv"))?;
        Ok(Self { norm, conv })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.conv.forward(&x)?;
        Ok(x)
    }
}

#[allow(unused)]
pub struct ConvNeXtBlock {
    dw_conv: Conv2d,
    norm: CustomLayerNorm,
    pw_conv1: Linear,
    pw_conv2: Linear,
}

#[allow(unused)]
impl ConvNeXtBlock {
    pub fn new(vb: VarBuilder, channel: usize, kernel_size: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: kernel_size / 2,
            stride: 1,
            groups: channel,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let dw_conv = conv2d(channel, channel, kernel_size, cfg, vb.pp("dw_conv"))?;
        let norm = CustomLayerNorm::new(vb.pp("norm"), channel, 1e-6, "channels_last")?;
        let pw_conv1 = linear(channel, channel * 4, vb.pp("pw_conv1"))?;
        let pw_conv2 = linear(4 * channel, channel, vb.pp("pw_conv2"))?;
        Ok(Self {
            dw_conv,
            norm,
            pw_conv1,
            pw_conv2,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();
        let x = self.dw_conv.forward(x)?;
        let x = x.permute((0, 2, 3, 1))?.contiguous()?; // (bs, c, h, w) -> (bs, h, w, c)
        let x = self.norm.forward(&x)?;
        let x = self.pw_conv1.forward(&x)?;
        let x = self.pw_conv2.forward(&x)?;
        let x = x.permute((0, 3, 1, 2))?.contiguous()?; // (bs, h, w, c) -> (bs, c, h, w)
        let x = x.add(&shortcut)?;
        Ok(x)
    }
}

#[allow(unused)]
pub struct ConvNeXtRes {
    blocks: Vec<ConvNeXtBlock>,
}

#[allow(unused)]
impl ConvNeXtRes {
    pub fn new(vb: VarBuilder, channel: usize, kernel_size: usize, depth: usize) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..depth {
            let block_i = ConvNeXtBlock::new(vb.pp(format!("block_{}", i)), channel, kernel_size)?;
            blocks.push(block_i);
        }
        Ok(Self { blocks })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

#[allow(unused)]
pub struct ConvNeXt {
    stem: Stem,
    res_blocks: Vec<ConvNeXtRes>,
    downsample_blocks: Vec<DownSample>,
    norm: CustomLayerNorm,
    head: Linear,
}

#[allow(unused)]
impl ConvNeXt {
    pub fn new(
        vb: VarBuilder,
        res_depth: Vec<usize>,
        res_channel: Vec<usize>,
        n_class: usize,
    ) -> Result<Self> {
        let stem = Stem::new(vb.pp("stem"), 3, res_channel[0], 4, 4)?;
        let mut res_blocks = Vec::new();
        for i in 0..4 {
            let res_i =
                ConvNeXtRes::new(vb.pp(format!("res_{}", i)), res_channel[i], 7, res_depth[i])?;
            res_blocks.push(res_i);
        }
        let mut downsample_blocks = Vec::new();
        for i in 0..3 {
            let downsample_i = DownSample::new(
                vb.pp(format!("downsample_{}", i)),
                res_channel[i],
                res_channel[i + 1],
                2,
                2,
            )?;
            downsample_blocks.push(downsample_i);
        }
        let norm = CustomLayerNorm::new(vb.pp("norm"), res_channel[3], 1e-6, "channels_last")?;
        let head = linear(res_channel[3], n_class, vb.pp("head"))?;
        Ok(Self {
            stem,
            res_blocks,
            downsample_blocks,
            norm,
            head,
        })
    }
}


#[allow(unused)]
impl CustomModule for ConvNeXt {
    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.stem.forward(x)?;
        let x = self.res_blocks[0].forward(&x)?;
        let x = self.downsample_blocks[0].forward(&x)?;
        let x = self.res_blocks[1].forward(&x)?;
        let x = self.downsample_blocks[1].forward(&x)?;
        let x = self.res_blocks[2].forward(&x)?;
        let x = self.downsample_blocks[2].forward(&x)?;
        let x = self.res_blocks[3].forward(&x)?;
        let x = x.mean(D::Minus1)?.mean(D::Minus1)?; // (bs, c, h, w) -> (bs, c, h) -> (bs, c)
        let x = self.norm.forward(&x)?;
        let x = self.head.forward(&x)?; // (bs, c) -> (bs, n_class)
        Ok(x)
    }
}


#[allow(unused)]
pub struct LRScheduler {
    current_step: usize,
    warmup_step: usize,
    total_step: usize,
    initial_lr: f64,
    final_lr: f64,
}

#[allow(unused)]
impl LRScheduler {
    pub fn new(
        warmup_step: usize,
        total_step: usize,
        initial_lr: f64,
        final_lr: f64,
    ) -> Result<Self> {
        if warmup_step >= total_step {
            return Err(Error::Msg(
                "warmup_step must be less than total_step".to_string(),
            ));
        }
        if initial_lr <= 0.0 || final_lr < 0.0 {
            return Err(Error::Msg("learning rates must be positive".to_string()));
        }
        Ok(Self {
            current_step: 0,
            warmup_step,
            total_step,
            initial_lr,
            final_lr,
        })
    }

    pub fn cosine_annealing(&mut self) -> Result<f64> {
        self.current_step += 1;
        if self.current_step > self.total_step {
            return Ok(self.final_lr);
        }
        let lr = if self.current_step <= self.warmup_step {
            // 预热阶段线性增长
            self.initial_lr * (self.current_step as f64 / self.warmup_step as f64)
        } else {
            // 余弦退火阶段
            let progress = (self.current_step - self.warmup_step) as f64
                / (self.total_step - self.warmup_step) as f64;
            self.final_lr
                + 0.5
                    * (self.initial_lr - self.final_lr)
                    * (1.0 + (progress * std::f64::consts::PI).cos())
        };
        Ok(lr)
    }
    pub fn reset_current_step(&mut self, total_step: usize, warmup_step: usize) {
        self.current_step = 0;
        self.total_step = total_step;
        self.warmup_step = warmup_step;
    }
}

#[allow(unused)]
pub fn label_smooth_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let bs = logits.dim(0)?;
    let log_probs = ops::log_softmax(logits, D::Minus1)?;
    let n_classes = logits.dim(D::Minus1)?;
    let targets = targets.to_dtype(DType::U32)?;
    let loss = log_probs
        .gather(&targets.unsqueeze(1)?, 1)?
        .squeeze(1)?
        .neg()?;
    let smooth_loss = log_probs
        .sum(D::Minus1)?
        .neg()?
        .affine(epsilon / n_classes as f64, 0.0)?;
    // （1-epsilon) * (-log p_y_true) + epsilon * (-sum log p_y_n) / n
    let loss = loss
        .affine(1.0 - epsilon, 0.0)?
        .add(&smooth_loss)?
        .sum_all()?
        .affine(1.0 / bs as f64, 0.)?;
    Ok(loss)
}

#[allow(unused)]
pub fn get_batch_loss<T: CustomModule>(model: &T, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let logits = model.forward(x, true)?;
    // let loss_ = loss::cross_entropy(&logits, &y)?;
    let loss_ = label_smooth_cross_entropy(&logits, y, 0.05)?;
    Ok(loss_)
}

#[allow(unused)]
pub fn get_batch_loss_acc<T: CustomModule>(
    model: &T,
    x: &Tensor,
    y: &Tensor,
) -> Result<(f32, f32)> {
    let logits = model.forward(x, false)?;
    let loss_ = loss::cross_entropy(&logits, &y)?;
    let loss_ = loss_.to_scalar::<f32>()?;
    let probs = ops::softmax(&logits, D::Minus1)?;
    let label = probs.argmax(1)?.to_dtype(DType::U8)?;
    let acc = label
        .eq(y)?
        .sum(0)?
        .to_dtype(candle_core::DType::F32)?
        .affine(1.0 / (y.dim(0)? as f64), 0.0)?;
    let acc = acc.to_scalar::<f32>()?;
    // println!("label: {}", label);
    // println!("true label: {}", y);
    Ok((loss_, acc))
}


#[allow(unused)]
pub fn get_loader_loss_acc<T: CustomModule>(
    model: &T,
    dataloader: &mut DataLoader,
) -> Result<(f32, f32)> {
    let mut loss_sum = 0.0;
    let mut acc_sum = 0.0;
    let mut count = 0;
    for batch in dataloader {
        let (x, y) = batch?;
        let (loss_, acc) = get_batch_loss_acc(model, &x, &y)?;
        loss_sum += loss_;
        acc_sum += acc;
        count += 1;
    }
    let loss_ = loss_sum / count as f32;
    let acc = acc_sum / count as f32;
    Ok((loss_, acc))
}

#[allow(unused)]
pub fn train_model<T: CustomModule>(
    model: &T,
    varmap: &VarMap,
    train_loader: &mut DataLoader,
    val_loader: &mut DataLoader,
    optimizer: &mut AdamW,
    scheduler: &mut LRScheduler,
    epochs: usize,
    eval_step: usize,
    n_class: usize,
    save_path: &str,
    model_type: &str,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut train_loss_vec = Vec::new();
    let mut val_loss_vec = Vec::new();
    let mut global_step = 0;
    let mut last_loss = 1000000000.0;
    let eval_save_path = format!("{}/eval_model_n_{}.safetensors", save_path, n_class);
    let last_save_path = format!("{}/last_model_n_{}.safetensors", save_path, n_class);
    let mut train_acc_vec = Vec::new();
    let mut val_acc_vec = Vec::new();
    let mut train_acc_last = 0.0;
    for epoch in 0..epochs {
        let _ = train_loader.reset()?;
        for batch in &mut *train_loader {
            let (x, y) = batch?;
            let loss_ = get_batch_loss(model, &x, &y)?;
            let _ = optimizer.backward_step(&loss_)?;
            let lr = scheduler.cosine_annealing()?;
            optimizer.set_learning_rate(lr);
            global_step += 1;
            if global_step % eval_step == 0 {
                let _ = val_loader.reset()?;
                let (val_loss, val_acc) = get_loader_loss_acc(model, val_loader)?;
                if val_loss < last_loss {
                    last_loss = val_loss;
                    varmap.save(&eval_save_path)?;
                }
                let train_loss = loss_.to_scalar::<f32>()?;
                println!(
                    "global_step: {} train_loss: {}, val_loss: {}, val_acc: {}",
                    global_step, train_loss, val_loss, val_acc
                );
                train_loss_vec.push(train_loss);
                val_loss_vec.push(val_loss);
                train_acc_vec.push(train_acc_last);
                val_acc_vec.push(val_acc);
            }
        }
        let _ = train_loader.reset()?;
        let _ = val_loader.reset()?;
        let (train_loss, train_acc) = get_loader_loss_acc(model, train_loader)?;
        train_acc_last = train_acc;
        let (val_loss, val_acc) = get_loader_loss_acc(model, val_loader)?;
        varmap.save(&last_save_path)?;
        println!(
            "epoch: {} train_loss: {}, val_loss: {}, train_acc: {}, val_acc: {}",
            epoch, train_loss, val_loss, train_acc_last, val_acc
        );
        train_loss_vec.push(train_loss);
        val_loss_vec.push(val_loss);
        train_acc_vec.push(train_acc_last);
        val_acc_vec.push(val_acc);
        plot_loss_curves(
            &train_loss_vec,
            &val_loss_vec,
            &format!("dog_loss_n_{}_{}.png", n_class, model_type),
            "loss curve",
        )?;
        plot_loss_curves(
            &train_acc_vec,
            &val_acc_vec,
            &format!("dog_acc_n_{}_{}.png", n_class, model_type),
            "acc curve",
        )?;
    }
    Ok((train_loss_vec, val_loss_vec))
}

#[allow(unused)]
pub fn train_img_main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let n_class = 10;
    let res_depth = [3, 3, 9, 3];
    let res_channel = [96, 192, 384, 768];
    let model_type = "convnext";
    let model = ConvNeXt::new(vb, res_depth.to_vec(), res_channel.to_vec(), n_class)?;
    // let model_type = "vit";
    // let model = ViT::new(vb, 224, 16, 1024, 10, 8, n_class)?;
    let data_group = ["train", "val"];
    let image_path = "/mnt/d/data/dog";
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), 1e-4)?;
    let batch_size = 60;
    let save_path = format!("weights/{}", model_type);
    let path = Path::new(&save_path);
    if !path.exists() {
        fs::create_dir_all(path).expect("Failed to create directory");
    }    
    let dataset_vec = get_img_dataset(
        image_path,
        data_group.to_vec(),
        n_class,
        &device,
        None,
        None,
    )?;
    assert_eq!(
        dataset_vec.len(),
        data_group.len(),
        "data_group and dataset_vec length must be equal"
    );
    let mut train_loader = DataLoader::new(dataset_vec[0].clone(), batch_size, true)?;
    let mut val_loader = DataLoader::new(dataset_vec[1].clone(), batch_size, false)?;
    let epochs = 50;
    let eval_step = 20;
    let total_step = epochs * (train_loader.len()? / batch_size);
    let warmup_step = total_step / 8;
    println!("total_step: {}, warmup_step: {}", total_step, warmup_step);
    let mut scheduler = LRScheduler::new(warmup_step, total_step, 5e-3, 1e-8)?;
    let (train_loss, val_loss) = train_model(
        &model,
        &varmap,
        &mut train_loader,
        &mut val_loader,
        &mut optimizer,
        &mut scheduler,
        epochs,
        eval_step,
        n_class,
        save_path.as_str(),
        model_type,
    )?;
    plot_loss_curves(
        &train_loss,
        &val_loss,
        &format!("dog_loss_n_{}_{}.png", n_class, model_type),
        "Loss Curves",
    )?;

    Ok(())
}
