#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>

#include <fstream>
#include <vector>
#include <iostream>
#include <cstdint>

// Function to display training data image in terminal
void show_image_in_terminal( vector<vector<double>> images,  vector<vector<double>> targets, int number){
    int count = 0;

    for (double ye : images[number]){
        count ++;
        if (ye>0.5){
            cout << "*" << " ";
        }
        else{
            cout << " " << " ";
        }
        if (count == 28){
            cout << endl;
            count = 0;
        }
    }

    for (int i=0 ; i<10 ; i++){
        cout << i << " : " << targets[number][i] << endl;
    }

    cout << endl;

}

// Function to reverse byte order (MNIST files are big-endian)
uint32_t reverse_int(uint32_t i) {
    uint8_t c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

// Function to read MNIST images
std::vector<std::vector<double>> read_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Cannot open file: " << filename << std::endl;
        return {};
    }

    uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    magic_number = reverse_int(magic_number);
    num_images = reverse_int(num_images);
    num_rows = reverse_int(num_rows);
    num_cols = reverse_int(num_cols);

    std::cout << "Number of images: " << num_images << std::endl;
    std::cout << "Image dimensions: " << num_rows << "x" << num_cols << std::endl;

    std::vector<std::vector<double>> images(num_images, std::vector<double>(num_rows * num_cols));
    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t j = 0; j < num_rows * num_cols; ++j) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            images[i][j] = static_cast<double>(temp) / 255.0; // Normalize to [0, 1]
        }
    }

    return images;
}

// Function to read MNIST labels
std::vector<std::vector<double>> read_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Cannot open file: " << filename << std::endl;
        return {};
    }

    uint32_t magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    magic_number = reverse_int(magic_number);
    num_items = reverse_int(num_items);

    std::cout << "Number of labels: " << num_items << std::endl;

    std::vector<std::vector<double>> labels(num_items, std::vector<double>(10, 0.0));
    for (uint32_t i = 0; i < num_items; ++i) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        labels[i][temp] = 1.0; // One-hot encoding
    }
    
    std::cout << " ---------------------------------------------------------------------------- \n" << std::endl;

    return labels;
}



std::vector<std::vector<double>> read_mnist_images(const std::string& filename);
std::vector<std::vector<double>> read_mnist_labels(const std::string& filename);

#endif // MNIST_LOADER_H