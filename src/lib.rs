use serde::{Serialize, Deserialize};
use std::ffi::{c_char, CStr, CString};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

unsafe extern "C" {
    pub fn train_bpe_ffi(text: *const c_char, vocab_size: i32);
    pub fn get_vocab_count() -> i32;
    pub fn get_vocab_item(index: i32) -> *const c_char;
    pub fn get_merge_count() -> i32;
    pub fn get_merge_pair(index: i32, first: *mut *const c_char, second: *mut *const c_char);
    pub fn free_string(ptr: *const c_char);
    pub fn cleanup_tokenizer();
}

pub const UNK_TOKEN: &str = "<unk>";
pub const PAD_TOKEN: &str = "<pad>";
pub const SOS_TOKEN: &str = "<sos>";
pub const EOS_TOKEN: &str = "<eos>";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BPEModel {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub token_to_id: HashMap<String, i32>,
    pub id_to_token: HashMap<i32, String>,
    pub special_tokens: Vec<String>,
}

impl BPEModel {
    fn build_mappings(vocab: &[String], special: &[String]) -> (HashMap<String, i32>, HashMap<i32, String>) {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut id = 0i32;
        for token in special {
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
            id += 1;
        }
        for token in vocab {
            if !token_to_id.contains_key(token) {
                token_to_id.insert(token.clone(), id);
                id_to_token.insert(id, token.clone());
                id += 1;
            }
        }
        (token_to_id, id_to_token)
    }

    pub fn save(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let model: BPEModel = serde_json::from_str(&json)?;
        Ok(model)
    }

    pub fn encode_to_ids(&self, text: &str) -> Vec<i32> {
        let tokens = encode(self, text);
        tokens.iter()
            .map(|t| *self.token_to_id.get(t).unwrap_or_else(|| self.token_to_id.get(UNK_TOKEN).unwrap()))
            .collect()
    }

    pub fn decode_from_ids(&self, ids: &[i32]) -> String {
        ids.iter()
            .map(|id| self.id_to_token.get(id).cloned().unwrap_or(UNK_TOKEN.to_string()))
            .collect()
    }

    pub fn pad_sequence(&self, ids: &mut Vec<i32>, target_len: usize) {
        let pad_id = *self.token_to_id.get(PAD_TOKEN).unwrap_or(&0);
        while ids.len() < target_len {
            ids.push(pad_id);
        }
    }

    pub fn truncate_sequence(&self, ids: &mut Vec<i32>, target_len: usize) {
        if ids.len() > target_len {
            ids.truncate(target_len);
        }
    }

    pub fn encode_with_special(&self, text: &str) -> Vec<i32> {
        let mut ids = vec![*self.token_to_id.get(SOS_TOKEN).unwrap_or(&0)];
        ids.extend(self.encode_to_ids(text));
        ids.push(*self.token_to_id.get(EOS_TOKEN).unwrap_or(&0));
        ids
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<i32>> {
        texts.iter().map(|t| self.encode_with_special(t)).collect()
    }

    pub fn pad_batch(&self, batch: &mut Vec<Vec<i32>>, target_len: usize) {
        for seq in batch.iter_mut() {
            self.pad_sequence(seq, target_len);
        }
    }

    pub fn vocab_coverage(&self, text: &str) -> f32 {
        let tokens = encode(self, text);
        let known = tokens.iter().filter(|t| self.token_to_id.contains_key(*t)).count();
        known as f32 / tokens.len() as f32
    }

    pub fn is_known_token(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    pub fn get_token_id(&self, token: &str) -> Option<i32> {
        self.token_to_id.get(token).copied()
    }
}

pub fn train(corpus_text: &str, vocab_size: i32) -> Result<BPEModel, Box<dyn std::error::Error>> {
    let c_text = CString::new(corpus_text)?;
    unsafe { train_bpe_ffi(c_text.as_ptr(), vocab_size); }

    let vocab_count = unsafe { get_vocab_count() };
    let mut vocab = Vec::new();
    for i in 0..vocab_count {
        let vocab_ptr = unsafe { get_vocab_item(i) };
        if !vocab_ptr.is_null() {
            if let Ok(vocab_str) = unsafe { CStr::from_ptr(vocab_ptr).to_str() } {
                vocab.push(vocab_str.to_string());
            }
            unsafe { free_string(vocab_ptr); }
        }
    }

    let merge_count = unsafe { get_merge_count() };
    let mut merges = Vec::new();
    for i in 0..merge_count {
        let mut first_ptr: *const c_char = std::ptr::null();
        let mut second_ptr: *const c_char = std::ptr::null();
        unsafe { get_merge_pair(i, &mut first_ptr, &mut second_ptr); }

        let first = if !first_ptr.is_null() {
            let s = unsafe { CStr::from_ptr(first_ptr).to_str().unwrap_or("").to_string() };
            unsafe { free_string(first_ptr); }
            s
        } else { String::new() };

        let second = if !second_ptr.is_null() {
            let s = unsafe { CStr::from_ptr(second_ptr).to_str().unwrap_or("").to_string() };
            unsafe { free_string(second_ptr); }
            s
        } else { String::new() };

        if !first.is_empty() && !second.is_empty() {
            merges.push((first, second));
        }
    }

    unsafe { cleanup_tokenizer(); }

    let special = vec![UNK_TOKEN.to_string(), PAD_TOKEN.to_string(), SOS_TOKEN.to_string(), EOS_TOKEN.to_string()];
    let (token_to_id, id_to_token) = BPEModel::build_mappings(&vocab, &special);

    Ok(BPEModel { vocab, merges, token_to_id, id_to_token, special_tokens: special })
}

pub fn encode(model: &BPEModel, text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
    for (first, second) in &model.merges {
        let merged = format!("{}{}", first, second);
        let mut new_tokens = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == *first && tokens[i + 1] == *second {
                new_tokens.push(merged.clone());
                i += 2;
            } else {
                new_tokens.push(tokens[i].clone());
                i += 1;
            }
        }
        tokens = new_tokens;
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_roundtrip() {
        let model = train("low lower lowest", 50).unwrap();
        let path = PathBuf::from("/tmp/test_model.json");
        model.save(&path).unwrap();
        let loaded = BPEModel::load(&path).unwrap();
        assert_eq!(model.vocab.len(), loaded.vocab.len());
    }

    #[test]
    fn test_batch_encoding() {
        let model = train("low lower lowest running happy", 50).unwrap();
        let texts = vec!["lower", "happy", "running"];
        let batch = model.encode_batch(&texts);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0][0], *model.token_to_id.get(SOS_TOKEN).unwrap());
        assert_eq!(batch[0][batch[0].len()-1], *model.token_to_id.get(EOS_TOKEN).unwrap());
    }

    #[test]
    fn test_pad_batch() {
        let model = train("a b c d e", 20).unwrap();
        let mut batch = model.encode_batch(&vec!["abc", "de"]);
        model.pad_batch(&mut batch, 10);
        assert_eq!(batch[0].len(), 10);
        assert_eq!(batch[1].len(), 10);
    }

    #[test]
    fn test_truncate_sequence() {
        let model = train("low lower lowest", 50).unwrap();
        let mut ids = model.encode_with_special("lowlowerlowest");
        let original_len = ids.len();
        model.truncate_sequence(&mut ids, 5);
        assert_eq!(ids.len(), 5.min(original_len));
    }

    #[test]
    fn test_vocab_coverage() {
        let model = train("low lower lowest", 50).unwrap();
        let coverage = model.vocab_coverage("lower");
        assert!(coverage > 0.0);
    }

    #[test]
    fn test_unknown_token_fallback() {
        let model = train("low lower lowest", 50).unwrap();
        let ids = model.encode_to_ids("xyzqwerty");
        let unk_id = *model.token_to_id.get(UNK_TOKEN).unwrap();
        assert!(ids.iter().all(|&id| id == unk_id || model.id_to_token.contains_key(&id)));
    }

    #[test]
    fn test_special_tokens_exist() {
        let model = train("low lower lowest", 50).unwrap();
        assert!(model.token_to_id.contains_key(UNK_TOKEN));
        assert!(model.token_to_id.contains_key(PAD_TOKEN));
    }

    #[test]
    fn test_encode_with_special() {
        let model = train("happy running", 50).unwrap();
        let ids = model.encode_with_special("happy");
        assert_eq!(ids[0], *model.token_to_id.get(SOS_TOKEN).unwrap());
        assert_eq!(ids[ids.len()-1], *model.token_to_id.get(EOS_TOKEN).unwrap());
    }

    #[test]
    fn test_subword_tokenization() {
        let model = train("low low low lower lower lowest", 50).unwrap();
        let tokens = encode(&model, "lower");
        assert!(tokens.contains(&"lo".to_string()) || tokens.contains(&"low".to_string()) || tokens == vec!["lower"]);
    }
}
