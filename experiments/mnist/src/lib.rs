use wasm_bindgen::prelude::*;

const RAW: &[u8] = include_bytes!("../artifacts/weights.bin");
const COUNT: usize = 17898;

#[wasm_bindgen]
pub struct Mnist {
    weights: Vec<f32>,
}

impl Default for Mnist {
    fn default() -> Self {
        assert_eq!(RAW.len(), COUNT * 4);
        Self {
            weights: RAW
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        }
    }
}

impl Mnist {
    fn forward(&self, pixels: &[f32]) -> Vec<f32> {
        let w = &self.weights;
        let mut tokens = [0.0_f32; 256];
        for t in 0..16 {
            for c in 0..16 {
                let mut sum = w[784 + c];
                for k in 0..49 {
                    let pixel = (t / 4 * 7 + k / 7) * 28 + t % 4 * 7 + k % 7;
                    sum += pixels[pixel] * w[c * 49 + k];
                }
                tokens[t * 16 + c] = sum.max(0.0);
            }
        }
        let mut hidden = [0.0_f32; 64];
        for j in 0..64 {
            let mut sum = w[17184 + j];
            for k in 0..256 {
                sum += tokens[k] * w[800 + j * 256 + k];
            }
            hidden[j] = sum.max(0.0);
        }
        (0..10)
            .map(|j| {
                let mut sum = w[17888 + j];
                for k in 0..64 {
                    sum += hidden[k] * w[17248 + j * 64 + k];
                }
                sum
            })
            .collect()
    }
}

#[wasm_bindgen]
impl Mnist {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Same embedded weights for CPU and custom WebGPU.
    pub fn weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    pub fn predict(&self, pixels: &[f32]) -> Result<Vec<f32>, JsValue> {
        if pixels.len() != 784
            || pixels
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(JsValue::from_str(
                "Expected 784 finite grayscale values in [0,1]",
            ));
        }
        Ok(self.forward(pixels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_pytorch_held_out_logits() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../artifacts/fixtures.json")).unwrap();
        let model = Mnist::default();
        for sample in fixtures.as_array().unwrap() {
            let input: Vec<f32> = sample["pixels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as f32 / 255.0)
                .collect();
            let actual = model.forward(&input);
            for (a, b) in actual.iter().zip(sample["logits"].as_array().unwrap()) {
                assert!(
                    (a - b.as_f64().unwrap() as f32).abs() < 0.0002,
                    "{a} vs {b}"
                );
            }
        }
    }
}
