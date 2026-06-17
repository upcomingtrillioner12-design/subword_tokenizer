#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

extern "C" {
    void train_bpe(const char* text, int vocab_size) {
        std::string corpus(text);
        std::cout << "Training BPE on corpus of size: " << corpus.size() << std::endl;
        std::cout << "Target vocab size: " << vocab_size << std::endl;
        // BPE algorithm will go here
    }
}
