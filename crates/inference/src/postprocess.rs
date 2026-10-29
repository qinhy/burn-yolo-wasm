//! YOLO11 NMS-free raw head -> detection boxes. Unit-testable without WebAssembly.
use serde::Serialize;

pub const SIZE: f32 = 320.0;
const MAX_CANDIDATES: usize = 1200;
const MAX_DETECTIONS: usize = 100;

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: usize,
}

impl Detection {
    fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let left = a.x1.max(b.x1);
    let top = a.y1.max(b.y1);
    let right = a.x2.min(b.x2);
    let bottom = a.y2.min(b.y2);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let union = a.area() + b.area() - intersection;
    if union <= 0.0 { 0.0 } else { intersection / union }
}

/// Expects [1, 4+num_classes, num_predictions] or [1, num_predictions, 4+num_classes].
/// Note: score is already the YOLO11 class confidence (no objectness multiplier).
pub fn decode_and_nms(
    data: &[f32], shape: [usize; 3], confidence: f32, iou_threshold: f32,
) -> Result<Vec<Detection>, String> {
    if shape[0] != 1 || data.len() != shape.iter().product::<usize>() {
        return Err(format!("Invalid YOLO tensor shape {:?} or data length", shape));
    }
    let channels_first = shape[1] >= 5 && shape[1] <= 204 && (shape[2] >= shape[1] || shape[2] < 5);
    let channels_last = !channels_first && shape[2] >= 5 && shape[2] <= 204 && (shape[1] >= shape[2] || shape[1] < 5);
    if !channels_first && !channels_last {
        return Err(format!("Expected raw YOLO [1,4+C,N] or [1,N,4+C], got {:?}", shape));
    }
    let (channels, predictions) = if channels_first {
        (shape[1], shape[2])
    } else {
        (shape[2], shape[1])
    };
    let num_classes = channels - 4;
    let at = |ch: usize, pred: usize| -> f32 {
        if channels_first { data[ch * predictions + pred] }
        else { data[pred * channels + ch] }
    };
    let mut candidates = Vec::new();
    for i in 0..predictions {
        let mut best_score = confidence;
        let mut best_class = None;
        for class in 0..num_classes {
            let score = at(class + 4, i);
            if score.is_finite() && score >= best_score {
                best_score = score;
                best_class = Some(class);
            }
        }
        let Some(class_id) = best_class else { continue; };
        let (cx, cy, w, h) = (at(0, i), at(1, i), at(2, i), at(3, i));
        if ![cx, cy, w, h].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
            continue;
        }
        let box_ = Detection {
            x1: (cx - w / 2.0).clamp(0.0, SIZE),
            y1: (cy - h / 2.0).clamp(0.0, SIZE),
            x2: (cx + w / 2.0).clamp(0.0, SIZE),
            y2: (cy + h / 2.0).clamp(0.0, SIZE),
            score: best_score,
            class_id,
        };
        if box_.area() > 0.0 { candidates.push(box_); }
    }

    candidates.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(MAX_CANDIDATES);
    let mut kept: Vec<Detection> = Vec::new();
    'outer: for candidate in candidates {
        for prev in &kept {
            if prev.class_id == candidate.class_id && iou(prev, &candidate) > iou_threshold {
                continue 'outer;
            }
        }
        kept.push(candidate);
        if kept.len() >= MAX_DETECTIONS { break; }
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suppress_same_class_but_not_different_class() {
        // [1, 6, 3] -> 2 classes, 3 predictions. Two overlapping class 0, one class 1.
        let tensor = [
            160., 160., 160., // center x
            160., 160., 160., // center y
            80., 80., 80.,    // width
            80., 80., 80.,    // height
            0.95, 0.90, 0.05, // class 0
            0.01, 0.02, 0.99, // class 1
        ];
        let found = decode_and_nms(&tensor, [1, 6, 3], 0.25, 0.45).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].class_id, 1);
        assert_eq!(found[1].class_id, 0);
    }

    #[test]
    fn rejects_invalid_shape() {
        assert!(decode_and_nms(&[0.], [1, 84, 2100], 0.25, 0.45).is_err());
    }
}
