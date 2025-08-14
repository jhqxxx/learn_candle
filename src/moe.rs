use crate::chapter3::FeedForward;
use crate::utils::net::{nonzero, onehot};
use candle_core::{D, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias, ops};

pub struct SparseMoeBlock {
    router: Linear,
    experts: Vec<FeedForward>,
    norm_topk_prob: bool,
    num_experts_per_token: usize,
}

impl SparseMoeBlock {
    pub fn new(
        vb: VarBuilder,
        num_experts: usize,
        in_dim: usize,
        hidden_dim: usize,
        norm_topk_prob: bool,
        num_experts_per_token: usize,
    ) -> Result<Self> {
        let router = linear_no_bias(in_dim, num_experts, vb.pp("router"))?;
        let mut experts = Vec::with_capacity(num_experts);
        let vb_e = vb.pp("experts");
        for i in 0..num_experts {
            let ffn = FeedForward::new(vb_e.pp(i), in_dim, hidden_dim, in_dim)?;
            experts.push(ffn);
        }
        Ok(Self {
            router,
            experts,
            norm_topk_prob,
            num_experts_per_token,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (bs, seq_len, embedding_dim) -> (bs*seq_len, embedding_dim)
        let (bs, seq_len, embedding_dim) = x.dims3()?;
        let x = x.reshape((bs * seq_len, embedding_dim))?;
        let router_weight = self.router.forward(&x)?; // (num_token, num_expters)
        let router_weight = ops::softmax_last_dim(&router_weight)?;
        let router_select = router_weight
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_token)?
            .contiguous()?;
        let mut router_weight = router_weight.gather(&router_select, D::Minus1)?;
        if self.norm_topk_prob {
            router_weight = router_weight.broadcast_div(&router_weight.sum_keepdim(D::Minus1)?)?;
        }
        let expert_mask = onehot(&router_select, self.experts.len())?
            .permute((2, 1, 0))?
            .to_dtype(candle_core::DType::U32)?;
        let expert_hit = expert_mask.sum((D::Minus1, D::Minus2))?;
        let expert_hit_vec = expert_hit.to_vec1::<u32>()?;
        let expert_hit_vec: Vec<usize> = expert_hit_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val > 0 { Some(i) } else { None })
            .collect();
        let mut final_x = x.zeros_like()?;
        for i in expert_hit_vec {
            let expert = &self.experts[i];
            let tokens = expert_mask.i(i)?;
            let (topk_id, token_id) = nonzero(&tokens)?;
            let token_id_tensor = Tensor::new(token_id.as_slice(), x.device())?;
            let select_tokens = x.index_select(&token_id_tensor, 0)?;
            let select_x = expert.forward(&select_tokens)?;
            let select_weight = router_weight.index_select(&token_id_tensor, 0)?.gather(
                &Tensor::new(topk_id.as_slice(), x.device())?.unsqueeze(D::Minus1)?,
                D::Minus1,
            )?;
            let select_x = select_x.broadcast_mul(&select_weight)?;
            final_x = final_x.index_add(&token_id_tensor, &select_x, 0)?;
        }
        final_x = final_x.reshape((bs, seq_len, embedding_dim))?;
        Ok(final_x)
    }
}
