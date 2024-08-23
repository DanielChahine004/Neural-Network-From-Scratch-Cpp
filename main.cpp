#include "NeuralNetwork.h"
#include "MNISTLoader.h"

using namespace std;


int main() {
    Neural_Network NN({784, 16, 16, 10});
    // Neural_Network NN({784, 2, 2, 10});
    // Neural_Network NN({784, 3, 3, 3, 3, 3, 3, 3, 3, 10});

    vector<vector<double>> training_data = read_mnist_images("train-images.idx3-ubyte");
    vector<vector<double>> targets = read_mnist_labels("train-labels.idx1-ubyte");


    const size_t temp_size = 500; 
    vector<vector<double>> temp_training_data(training_data.begin(), training_data.begin() + temp_size);
    vector<vector<double>> temp_targets(targets.begin(), targets.begin() + temp_size);


    int index = 1039;

    Eigen::MatrixXd output_before_training = predict_output_with_neural_network(&NN, training_data[index]);

    train_neural_network(&NN, temp_training_data, temp_targets, 15, 1);

    Eigen::MatrixXd output_after_training = predict_output_with_neural_network(&NN, training_data[index]);

    show_image_in_terminal(training_data, targets, index);

    cout << output_before_training << endl << endl;
    cout << output_after_training << endl << endl;

    

    return 0;
}