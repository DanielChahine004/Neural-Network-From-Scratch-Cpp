#include <iostream>
#include <chrono>
#include <thread>

#include "NeuralNetwork.h"
#include "MNISTLoader.h"
#include "GuiMaker.h"

using namespace std;


int main() {
    Neural_Network NN({784, 16, 16, 10});

    // Neural_Network NN = load_neural_network("2024-09-07 MNIST NN");

    // Neural_Network NN({784, 2, 2, 10});
    // Neural_Network NN({784, 3, 3, 3, 3, 3, 3, 3, 3, 10});

    vector<vector<double>> training_data = read_mnist_images("train-images.idx3-ubyte");
    vector<vector<double>> training_targets = read_mnist_labels("train-labels.idx1-ubyte");

    vector<vector<double>> validation_data = read_mnist_images("t10k-images.idx3-ubyte");
    vector<vector<double>> validation_targets = read_mnist_labels("t10k-labels.idx1-ubyte");


    const size_t temp_size = 6000; 
    vector<vector<double>> temp_training_data(training_data.begin(), training_data.begin() + temp_size);
    vector<vector<double>> temp_targets(training_targets.begin(), training_targets.begin() + temp_size);



    int index = 1043;

    // Eigen::MatrixXd output_before_training = predict_output_with_neural_network(&NN, training_data[index]);

    // show_neural_network(&NN);
    
    // Your main loop or other operations here
    GUIMaker gui(L"Neural Network Visualization", &NN);
    gui.Initialize();

    train_neural_network(&NN, temp_training_data, temp_targets, 3, 1);
    // save_neural_network(&NN, "2024-09-07 MNIST NN");

    // show_image_in_terminal(training_data, training_targets, index);

    // forward_pass(&NN, training_data[index]);

    // show_neural_network(&NN);


    // Eigen::MatrixXd output_after_training = predict_output_with_neural_network(&NN, training_data[index]);
    // show_image_in_terminal(training_data, training_targets, index);
    // cout << output_before_training << endl << endl;
    // cout << output_after_training << endl << endl;


    // double accuracy = calculate_network_accuracy(&NN, validation_data, validation_targets);
    // cout << accuracy << "% of the dataset was correctly identified" << endl;

    return 0;
}