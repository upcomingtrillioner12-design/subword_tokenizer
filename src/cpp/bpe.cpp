#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <set>

using namespace std;

class BPETokenizer {
public:
    set<string> vocab;
    vector<pair<string, string>> merges;
    int vocab_size;

    BPETokenizer(int size) : vocab_size(size) {}

    map<pair<string, string>, int> get_stats(const vector<string>& tokens) {
        map<pair<string, string>, int> stats;
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            stats[{tokens[i], tokens[i+1]}]++;
        }
        return stats;
    }

    vector<string> merge_pair(const vector<string>& tokens, const string& pair1, const string& pair2) {
        vector<string> result;
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i + 1 < tokens.size() && tokens[i] == pair1 && tokens[i+1] == pair2) {
                result.push_back(pair1 + pair2);
                i++;
            } else {
                result.push_back(tokens[i]);
            }
        }
        return result;
    }

    void train(const string& text) {
        vector<string> words;
        string current;
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!current.empty()) { words.push_back(current); current.clear(); }
            } else {
                current += c;
            }
        }
        if (!current.empty()) words.push_back(current);

        vector<vector<string>> word_tokens;
        for (const auto& word : words) {
            vector<string> chars;
            for (char c : word) {
                chars.push_back(string(1, c));
                vocab.insert(string(1, c));
            }
            word_tokens.push_back(chars);
        }

        int num_merges = min(vocab_size - (int)vocab.size(), 50);

        for (int merge = 0; merge < num_merges; merge++) {
            map<pair<string, string>, int> stats;
            for (const auto& tokens : word_tokens) {
                for (size_t i = 0; i < tokens.size() - 1; i++) {
                    stats[{tokens[i], tokens[i+1]}]++;
                }
            }

            if (stats.empty()) break;

            auto best_pair = stats.begin();
            for (auto it = stats.begin(); it != stats.end(); ++it) {
                if (it->second > best_pair->second) best_pair = it;
            }

            string pair1 = best_pair->first.first;
            string pair2 = best_pair->first.second;
            string merged = pair1 + pair2;

            for (auto& tokens : word_tokens) {
                tokens = merge_pair(tokens, pair1, pair2);
            }

            merges.push_back({pair1, pair2});
            vocab.insert(merged);
        }
    }

    vector<string> tokenize_word(const string& word) {
        vector<string> tokens;
        for (char c : word) tokens.push_back(string(1, c));

        for (const auto& merge : merges) {
            vector<string> new_tokens;
            size_t i = 0;
            while (i < tokens.size()) {
                if (i + 1 < tokens.size() && tokens[i] == merge.first && tokens[i+1] == merge.second) {
                    new_tokens.push_back(merge.first + merge.second);
                    i += 2;
                } else {
                    new_tokens.push_back(tokens[i]);
                    i++;
                }
            }
            tokens = new_tokens;
        }
        return tokens;
    }

    string encode(const string& text) {
        vector<string> all_tokens;
        string current_word;

        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!current_word.empty()) {
                    auto word_tokens = tokenize_word(current_word);
                    all_tokens.insert(all_tokens.end(), word_tokens.begin(), word_tokens.end());
                    current_word.clear();
                }
                all_tokens.push_back(string(1, c));
            } else {
                current_word += c;
            }
        }

        if (!current_word.empty()) {
            auto word_tokens = tokenize_word(current_word);
            all_tokens.insert(all_tokens.end(), word_tokens.begin(), word_tokens.end());
        }

        string result;
        for (size_t i = 0; i < all_tokens.size(); i++) {
            result += all_tokens[i];
            if (i < all_tokens.size() - 1) result += ", ";
        }
        return result;
    }

    vector<string> get_vocab_list() {
        return vector<string>(vocab.begin(), vocab.end());
    }

    vector<pair<string, string>> get_merges() {
        return merges;
    }
};

static BPETokenizer* g_tokenizer = nullptr;
static vector<string> g_vocab_cache;
static vector<pair<string, string>> g_merges_cache;

extern "C" {

    void train_bpe_ffi(const char* text, int vocab_size) {
        if (g_tokenizer) {
            delete g_tokenizer;
        }
        g_tokenizer = new BPETokenizer(vocab_size);
        g_tokenizer->train(string(text));

        g_vocab_cache = g_tokenizer->get_vocab_list();
        g_merges_cache = g_tokenizer->get_merges();
    }

    int get_vocab_count() {
        return (int)g_vocab_cache.size();
    }

    const char* get_vocab_item(int index) {
        if (index < 0 || index >= (int)g_vocab_cache.size()) {
            return nullptr;
        }
        char* cstr = new char[g_vocab_cache[index].length() + 1];
        strcpy(cstr, g_vocab_cache[index].c_str());
        return cstr;
    }

    int get_merge_count() {
        return (int)g_merges_cache.size();
    }

    void get_merge_pair(int index, const char** first, const char** second) {
        if (index < 0 || index >= (int)g_merges_cache.size()) {
            *first = nullptr;
            *second = nullptr;
            return;
        }
        char* f = new char[g_merges_cache[index].first.length() + 1];
        char* s = new char[g_merges_cache[index].second.length() + 1];
        strcpy(f, g_merges_cache[index].first.c_str());
        strcpy(s, g_merges_cache[index].second.c_str());
        *first = f;
        *second = s;
    }

    void free_string(const char* ptr) {
        delete[] ptr;
    }

    void cleanup_tokenizer() {
        if (g_tokenizer) {
            delete g_tokenizer;
            g_tokenizer = nullptr;
        }
        g_vocab_cache.clear();
        g_merges_cache.clear();
    }
}
