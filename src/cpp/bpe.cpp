#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <utility>
#include <set>

typedef std::pair<std::string, std::string> StringPair;

// Custom hash for std::pair<std::string, std::string>
struct PairHash {
    size_t operator()(const StringPair& p) const {
        return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
    }
};

extern "C" {
    void train_bpe(const char* text, int vocab_size) {
        std::string corpus(text);
        
        // Initialize: split corpus into characters
        std::vector<std::string> tokens;
        for (char c : corpus) {
            tokens.push_back(std::string(1, c));
        }
        
        int initial_vocab = 256; // Assuming ASCII
        int current_vocab = initial_vocab;
        int merge_count = 0;
        std::set<StringPair> merged_pairs; // Track which pairs have been merged
        
        std::cout << "\n=== BPE Training Start ===" << std::endl;
        std::cout << "Corpus size: " << corpus.size() << " chars" << std::endl;
        std::cout << "Initial vocab size: " << initial_vocab << std::endl;
        std::cout << "Target vocab size: " << vocab_size << std::endl;
        std::cout << "Initial tokens: " << tokens.size() << std::endl;
        
        // Iteratively merge pairs
        while (current_vocab < vocab_size) {
            // Count pair frequencies
            std::unordered_map<StringPair, int, PairHash> pair_freq;
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                StringPair pair = {tokens[i], tokens[i + 1]};
                pair_freq[pair]++;
            }
            
            if (pair_freq.empty()) {
                std::cout << "No more pairs to merge. Stopping at vocab size: " << current_vocab << std::endl;
                break;
            }
            
            // Find most frequent pair
            StringPair most_frequent = pair_freq.begin()->first;
            int max_freq = pair_freq.begin()->second;
            
            for (const auto& entry : pair_freq) {
                if (entry.second > max_freq) {
                    max_freq = entry.second;
                    most_frequent = entry.first;
                }
            }
            
            // Skip if pair is too long (prevents merging into huge strings)
            std::string merged_token = most_frequent.first + most_frequent.second;
            if (merged_token.length() > 20) {
                // Skip this pair and move to next most frequent
                int second_max = 0;
                StringPair second_frequent = most_frequent;
                for (const auto& entry : pair_freq) {
                    if (entry.first != most_frequent && entry.second > second_max) {
                        second_max = entry.second;
                        second_frequent = entry.first;
                    }
                }
                if (second_max == 0) {
                    std::cout << "No valid pairs to merge (all too long). Stopping at vocab size: " << current_vocab << std::endl;
                    break;
                }
                most_frequent = second_frequent;
                max_freq = second_max;
                merged_token = most_frequent.first + most_frequent.second;
            }
            
            // Merge most frequent pair throughout tokens
            std::vector<std::string> new_tokens;
            
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i + 1 < tokens.size() && 
                    tokens[i] == most_frequent.first && 
                    tokens[i + 1] == most_frequent.second) {
                    new_tokens.push_back(merged_token);
                    ++i; // Skip next token since we merged it
                } else {
                    new_tokens.push_back(tokens[i]);
                }
            }
            
            tokens = new_tokens;
            merged_pairs.insert(most_frequent);
            current_vocab++;
            merge_count++;
            
            // Log progress every 5 merges
            if (merge_count % 5 == 0) {
                std::cout << "Merge #" << merge_count << ": ('" 
                          << most_frequent.first << "' + '" 
                          << most_frequent.second << "' -> '" 
                          << merged_token << "') freq=" << max_freq 
                          << ", vocab=" << current_vocab 
                          << ", tokens=" << tokens.size() << std::endl;
            }
        }
        
        std::cout << "\n=== BPE Training Complete ===" << std::endl;
        std::cout << "Total merges performed: " << merge_count << std::endl;
        std::cout << "Final vocab size: " << current_vocab << std::endl;
        std::cout << "Final token sequence length: " << tokens.size() << std::endl;
        std::cout << "Compression ratio: " << static_cast<double>(corpus.size()) / tokens.size() 
                  << "x (original chars / final tokens)" << std::endl;
        
        // Show final tokens (first 30 for brevity)
        std::cout << "\nFinal tokenization (first 30 tokens):" << std::endl;
        std::cout << "[ ";
        for (size_t i = 0; i < std::min(size_t(30), tokens.size()); ++i) {
            if (tokens[i].length() <= 1) {
                std::cout << "'" << tokens[i] << "' ";
            } else {
                std::cout << "[" << tokens[i] << "] ";
            }
        }
        if (tokens.size() > 30) {
            std::cout << "... (" << tokens.size() - 30 << " more)";
        }
        std::cout << " ]" << std::endl;
    }
}
