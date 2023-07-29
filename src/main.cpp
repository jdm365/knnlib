#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <omp.h>
#include <unordered_map>
#include <algorithm>


std::vector<std::vector<std::string>> shingle(std::vector<std::string> &data, std::unordered_map<std::string, unsigned int>& vocab, int k) {
	std::vector<std::vector<std::string>> shingles;

	// Iterate over all documents
	for (int idx = 0; idx < data.size(); ++idx) {
		std::vector<std::string> shingle;
		// Iterate over all k-grams
		for (int idx2 = 0; idx2 < data[idx].size() - k + 1; ++idx2) {
			std::string kgram = data[idx].substr(idx2, k);
			// If k-gram is not in vocab, add it
			if (vocab.find(kgram) == vocab.end()) {
				vocab[kgram] = vocab.size();
			}
			shingle.push_back(kgram);
		}
		shingles.push_back(shingle);
	}

	return shingles;
}

char hash_string(const std::string& str) {
	char hash = 0;
	for (char ch : str) {
		hash = ((hash << 5) + hash) + ch;
	}
	return hash % 4;
}

std::vector<char> hash_dataset(std::vector<std::string> &string_data, int n_hashes) {
	std::unordered_map<std::string, unsigned int> vocab;
	std::vector<std::vector<std::string>> shingles = shingle(string_data, vocab, 4);

	// Hash into 16 buckets represented by 4 bits
	std::vector<char> signatures(n_hashes * shingles.size() / 2);

	#pragma omp parallel for
	for (int idx = 0; idx < n_hashes; ++idx) {
		int rand_int = rand() % 1024;

		char hash_bit_repr = 0;
		for (int jdx = 0; jdx < (int)shingles.size(); ++jdx) {
			char min_hash = 16;

			for (int kdx = 0; kdx < (int)shingles[jdx].size(); ++kdx) {
				min_hash = std::min(hash_string(shingles[jdx][kdx]), min_hash);
			}
			
			if (jdx % 2 == 0) {
				hash_bit_repr = min_hash;
			} 
			else {
				hash_bit_repr = (hash_bit_repr << 4) + min_hash;
				signatures[idx * 8 + (jdx / 2)] = hash_bit_repr;
			}
		}
	}

	return signatures;
}


int main() {
	// Generate 1000000 random strings of length 25
	std::vector<std::string> data;
	for (int idx = 0; idx < 1000000; ++idx) {
		std::string random_string = "";
		for (int idx2 = 0; idx2 < 25; ++idx2) {
			random_string += (char) (rand() % 26 + 97);
		}
		data.push_back(random_string);
	}

	std::vector<char> signatures = hash_dataset(data, 128);
	std::cout << "Number of signatures: " << signatures.size() << std::endl;

	return 0;
}
