use crate::impl_evo_fp::smarts_fingerprint::SmartsFingerprint;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Member {
    pub fingerprint: SmartsFingerprint,
    pub metric: Option<f32>,
}

impl Member {
    pub fn new(fingerprint: SmartsFingerprint) -> Member {
        Member {
            fingerprint,
            metric: None,
        }
    }
}
