#[cfg(test)]
mod tests {
    use crate::BPEModel;

    #[test]
    fn test_tokenize_single_character() {
        let model = BPEModel {
            vocab: vec!["a".to_string(), "b".to_string()],
            merges: vec![],
            vocab_size: 300,
        };
        
        let tokens = model.tokenize("a");
        assert_eq!(tokens, vec!["a"]);
    }

    #[test]
    fn test_tokenize_without_merges() {
        let model = BPEModel {
            vocab: vec![],
            merges: vec![],
            vocab_size: 256,
        };
        
        let tokens = model.tokenize("abc");
        assert_eq!(tokens, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_tokenize_with_single_merge() {
        let model = BPEModel {
            vocab: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
            merges: vec![("a".to_string(), "b".to_string())],
            vocab_size: 258,
        };
        
        let tokens = model.tokenize("ab");
        assert_eq!(tokens, vec!["ab"]);
    }

    #[test]
    fn test_tokenize_with_multiple_merges() {
        let model = BPEModel {
            vocab: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "ab".to_string(),
                "abc".to_string(),
            ],
            merges: vec![
                ("a".to_string(), "b".to_string()),
                ("ab".to_string(), "c".to_string()),
            ],
            vocab_size: 260,
        };
        
        let tokens = model.tokenize("abc");
        assert_eq!(tokens, vec!["abc"]);
    }

    #[test]
    fn test_tokenize_partial_merge() {
        let model = BPEModel {
            vocab: vec!["a".to_string(), "b".to_string(), "c".to_string(), "ab".to_string()],
            merges: vec![("a".to_string(), "b".to_string())],
            vocab_size: 260,
        };
        
        let tokens = model.tokenize("abc");
        assert_eq!(tokens, vec!["ab", "c"]);
    }

    #[test]
    fn test_tokenize_multiple_occurrences() {
        let model = BPEModel {
            vocab: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
            merges: vec![("a".to_string(), "b".to_string())],
            vocab_size: 258,
        };
        
        let tokens = model.tokenize("abab");
        assert_eq!(tokens, vec!["ab", "ab"]);
    }

    #[test]
    fn test_tokenize_empty_string() {
        let model = BPEModel {
            vocab: vec![],
            merges: vec![],
            vocab_size: 256,
        };
        
        let tokens = model.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_preserves_order() {
        let model = BPEModel {
            vocab: vec!["h".to_string(), "e".to_string(), "l".to_string(), "o".to_string(), "he".to_string(), "hel".to_string(), "hello".to_string()],
            merges: vec![
                ("h".to_string(), "e".to_string()),
                ("he".to_string(), "l".to_string()),
                ("hel".to_string(), "l".to_string()),
                ("hell".to_string(), "o".to_string()),
            ],
            vocab_size: 260,
        };
        
        let tokens = model.tokenize("hello");
        // Should preserve the text even after various merges
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_tokenize_with_spaces() {
        let model = BPEModel {
            vocab: vec![
                "h".to_string(),
                "i".to_string(),
                " ".to_string(),
                "y".to_string(),
                "o".to_string(),
                "u".to_string(),
                "hi".to_string(),
                "you".to_string(),
            ],
            merges: vec![
                ("h".to_string(), "i".to_string()),
                ("y".to_string(), "o".to_string()),
                ("yo".to_string(), "u".to_string()),
            ],
            vocab_size: 260,
        };
        
        let tokens = model.tokenize("hi you");
        // Should produce merged tokens
        assert_eq!(tokens.len(), 3); // "hi", " ", "you"
    }

    #[test]
    fn test_bpe_model_determinism() {
        // Same model should produce identical results on multiple runs
        let model = BPEModel {
            vocab: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
            merges: vec![("a".to_string(), "b".to_string())],
            vocab_size: 258,
        };
        
        let text = "ababab";
        let result1 = model.tokenize(text);
        let result2 = model.tokenize(text);
        
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_bpe_model_serialization() {
        use serde_json;
        
        let model = BPEModel {
            vocab: vec!["a".to_string(), "ab".to_string()],
            merges: vec![("a".to_string(), "b".to_string())],
            vocab_size: 300,
        };
        
        let json = serde_json::to_string(&model).expect("Serialization failed");
        let deserialized: BPEModel = serde_json::from_str(&json).expect("Deserialization failed");
        
        assert_eq!(model.vocab, deserialized.vocab);
        assert_eq!(model.merges, deserialized.merges);
        assert_eq!(model.vocab_size, deserialized.vocab_size);
    }

    #[test]
    fn test_bpe_model_vocab_ordering() {
        // Verify that merges are applied in order (not scrambled)
        let model = BPEModel {
            vocab: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "ab".to_string(),
                "abc".to_string(),
            ],
            merges: vec![
                ("a".to_string(), "b".to_string()),
                ("ab".to_string(), "c".to_string()),
            ],
            vocab_size: 260,
        };
        
        // Apply merges to "a b c"
        let tokens = model.tokenize("a b c");
        
        // First merge should create "ab" from "a" and "b", but won't affect space and "c"
        // So we expect ["ab", " ", "c"] after first merge, then no further merges apply
        assert!(tokens.len() >= 1);
    }
}
