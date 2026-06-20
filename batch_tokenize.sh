#!/bin/bash

MODEL="model.json"

test_texts=(
    "hello world how are you doing today"
    "artificial intelligence and machine learning are transforming technology"
    "the quick brown fox jumps over the lazy dog"
    "i love programming in rust and python"
    "neural networks can process complex patterns in data"
    "good morning have a wonderful day ahead"
    "please help me understand this concept better"
    "we need to finish this project before deadline"
    "the weather is nice today lets go outside"
    "thank you so much for your help and support"
    "i am feeling happy and excited about this"
    "can you explain how this works step by step"
    "she bought a new car last week"
    "they went to the beach during summer vacation"
    "he has been working hard all day long"
    "the children are playing in the garden"
    "my favorite food is pizza and pasta"
    "we watched a great movie last night"
    "the doctor said i need more rest"
    "students must complete their assignments on time"
)

echo "=== BATCH TOKENIZATION ==="
echo "Model: $MODEL"
echo "Tests: ${#test_texts[@]}"
echo ""

total_tokens=0
total_time=0

for i in "${!test_texts[@]}"; do
    text="${test_texts[$i]}"
    echo "--- Test $((i+1)) ---"
    echo "Text: '$text'"
    
    # Run and capture output
    start=$(date +%s%N)
    output=$(cargo run -- --model "$MODEL" --text "$text" 2>/dev/null)
    end=$(date +%s%N)
    
    # Extract token count from IDs line
    ids_line=$(echo "$output" | grep "IDs:" | sed 's/IDs:.*\[//;s/\]//')
    token_count=$(echo "$ids_line" | tr ',' '\n' | wc -l)
    total_tokens=$((total_tokens + token_count))
    
    # Time in ms
    elapsed=$(( (end - start) / 1000000 ))
    total_time=$((total_time + elapsed))
    
    echo "Tokens: $token_count | Time: ${elapsed}ms"
    echo ""
done

echo "=== SUMMARY ==="
echo "Total tests: ${#test_texts[@]}"
echo "Total tokens generated: $total_tokens"
echo "Total time: ${total_time}ms"
echo "Avg time per test: $((total_time / ${#test_texts[@]}))ms"
echo "Avg tokens per test: $((total_tokens / ${#test_texts[@]}))"
