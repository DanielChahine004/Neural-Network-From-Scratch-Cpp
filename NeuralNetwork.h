#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include "Visualiser.h"

using namespace std;

struct Neural_Network {

    vector<Eigen::MatrixXd> connection_layers; // a list of 2d matrixes sized (# nodes in next layer, # nodes in current layer)
    vector<Eigen::MatrixXd> neuron_layers; // a list of column matrixes sized (# nodes in current later, 1)
    vector<Eigen::MatrixXd> bias_layers; // a list of column matrixes sized (# nodes in current later, 1)

    Neural_Network(const vector<int>& NN_layers){
        // Set up random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        // std::normal_distribution<double> distribution(0.0, 1.0 / sqrt(NN_layers[0]));  // Xavier initialization


        // Initialize connection_layers and bias_layers
        neuron_layers.reserve(NN_layers.size());
        for (int i = 0; i < NN_layers.size(); i++) { // i goes 0, 1, 2, 3
            neuron_layers.push_back(Eigen::MatrixXd::Zero(NN_layers[i], 1));
        }

        // Initialize connection_layers and bias_layers
        for (int i = 0; i < NN_layers.size() - 1; i++) { // i goes 0, 1, 2

            Eigen::MatrixXd connection_matrix = Eigen::MatrixXd::Zero(NN_layers[i+1], NN_layers[i]);
            for (int right = 0; right < connection_matrix.rows(); right++) {
                for (int left = 0; left < connection_matrix.cols(); left++) {
                    // connection_matrix(right, left) = 0.95;
                    connection_matrix(right, left) = distribution(generator);
                }
            }
            connection_layers.push_back(connection_matrix);

            Eigen::MatrixXd bias_matrix = Eigen::MatrixXd::Zero(NN_layers[i+1], 1);
            for (int neuron = 0; neuron < bias_matrix.rows(); neuron++) {
                // bias_matrix(neuron, 0) = 0;
                bias_matrix(neuron, 0) = distribution(generator);
            }
            bias_layers.push_back(bias_matrix);
        }

        cout << " ----------------------------------------------------------------------------" << endl;
        cout << "Neural Network Structure : ";
        for (auto neuron : neuron_layers){
            cout << neuron.rows() << " -> ";
        }
        cout << endl << endl;
    }
};

// Sigmoid activation function for scalar values
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Apply sigmoid activation to an Eigen matrix
void apply_sigmoid_activation(Eigen::MatrixXd& matrix) {
    matrix = matrix.unaryExpr(&sigmoid);
}

Eigen::MatrixXd calculate_loss_matrix(Eigen::MatrixXd outputs_matrix, const vector<double> label){ // calculates the loss of the outputs (differences between desired and predicted, squared)
    Eigen::Map<const Eigen::MatrixXd> label_matrix (&label[0], label.size(), 1); // makes the labels a matrix for efficient calculation of costs
    // Eigen::MatrixXd costs = pow(label_matrix.array() - outputs_matrix.array(), 2); // (y(label) - activation(output))^2
    Eigen::MatrixXd costs = (label_matrix.array() - outputs_matrix.array()).square() / outputs_matrix.rows();  // normalized loss

    return costs;
}

Eigen::MatrixXd forward_pass(Neural_Network* NN, vector<double> entry){  // uses the input activations and weights, and biases to calculate activations of neuron layers; forward propagation

    NN->neuron_layers[0] = Eigen::Map<const Eigen::MatrixXd>(&entry[0], entry.size(), 1); // input example entry into first layer of NN
    
    Eigen::MatrixXd result_matrix; // placeholder for the result matrix for the next layer (AW+B)

    for (int L=0 ; L<NN->connection_layers.size() ; L++){ // L goes from 0, to 1, to 2 - go through the rest of the neuron layers

        result_matrix = NN->connection_layers[L] * NN->neuron_layers[L] + NN->bias_layers[L]; // the result matrix = Activations * Weights + Bias
        apply_sigmoid_activation(result_matrix); // apply the sigmoid function to constrain activations between 0 and 1
        NN->neuron_layers[L+1] = result_matrix; // give the next layers neurons their calculated activations
    }

    return result_matrix;
}



void back_propagate(Neural_Network* NN, const Eigen::MatrixXd loss_matrix, const std::vector<double> labels, double learning_rate) { // the big fucker

    // Initialize DConDWs as a matrix list of equal size and dimensions as NN->connection_layers to hold the gradients of each weight in the network
    std::vector<Eigen::MatrixXd> DConDWs; // makes a list of matrixes the same size as connection_layers so each weight has a corresponding delta
    DConDWs.reserve(NN->connection_layers.size()); // reserve the same number of entities as connection_layers; every connection_layer matrix has an equally-sized corresponding DConDWs matrix
    for (const auto& layer : NN->connection_layers) {
        DConDWs.push_back(Eigen::MatrixXd::Zero(layer.rows(), layer.cols())); // DConDWs initialise as Zeros... maybe a problem, maybe not
    }

    // Initialise all_neuron_deltas as a matrix list of equal size and dimensions as NN->neuron_layers to hold the delta value of each neuron in the network
    std::vector<Eigen::MatrixXd> all_neuron_deltas; // list of the delta values for all the neurons EXCEPT THE INPUT LAYER NEURONS... all_neuron_deltas is the SAME dimensions and size as biases
    all_neuron_deltas.reserve(NN->bias_layers.size()); // reserve the same number of entities as bias_layers; every neuron (except inputs) get a delta calculated for them
    for (const auto& layer : NN->bias_layers) {
        all_neuron_deltas.push_back(Eigen::MatrixXd::Zero(layer.rows(), 1)); // Deltas initialise as Zeros... maybe a problem, maybe not
    }

    // *************************** the output delta value for each neuron IS the DWonDB for that neuron!! *****************************************************

    // Sets the deltas of the last layers neurons ---------- 7:33 Backpropagation Algorithm | Neural Networks video
    for (int neuron = 0; neuron < NN->neuron_layers.back().rows(); neuron++) { // neuron goes 0, 1, 2, ... 9
        double AonZ = NN->neuron_layers.back()(neuron, 0)  *  (  1  -  NN->neuron_layers.back()(neuron, 0)  ); // AonZ = a(L) * (1-a(L))
        double ConA = 2 * (labels[neuron] - NN->neuron_layers.back()(neuron, 0)); // ConA = 2(desired output for that neuron - that neurons activation)
        all_neuron_deltas.back()(neuron, 0) = ConA * AonZ; // populates the output deltas in the output layer of the all_neuron_deltas matrix
    }
    
    // Sets the gradient of the weights (aka connections) (DConDW) feeding into the outputs layer; populates the DConDws structure ---- 7:21 Backpropagation Algorithm | Neural Networks video
    for (int left=0 ; left<NN->neuron_layers[NN->neuron_layers.size()-2].rows() ; left++){ // left iterates through the neurons of the second last neurons layer
        for (int right=0 ; right<NN->neuron_layers.back().rows() ; right++){ // right iterates through the neurons of the output layer
            double activation_of_left_neuron = NN->neuron_layers[NN->neuron_layers.size()-2](left, 0); // activation of the left side neuron (for each iteration)
            DConDWs.back()(right,left) = all_neuron_deltas.back()(right,0) * activation_of_left_neuron; // the delta of the right (output) side neuron, times the activation of the left side (last hidden layer) neuron, is the DConDW of the weight between those two neurons
        }
    }

    // Back-propagates the delta values of all the neurons in the network, populating the all_neuron_deltas for the network (except input layer neurons of course) ()
    for (int L = NN->neuron_layers.size()-2 ; L>0 ; L--){ // L goes 2, 1
        for (int left=0 ; left<NN->neuron_layers[L].rows() ; left++){ // left iterates the left side neurons 
            double delta = 0; // for each neuron on the left side, delta will hold the sum of : the deltas of the right side neurons, times, the weight between the left neuron we are iterating on, and each of the right side neurons, times, A(1-A) where A is the activation of the left side neuron (which is the sigmoid derivitive of the activation of the left side neuron)
            
            for (int right=0 ; right<NN->neuron_layers[L+1].rows() ; right++){ // right iterates the right side neurons
                delta += all_neuron_deltas[L](right,0)  *  NN->connection_layers[L](right, left)  *  ( NN->neuron_layers[L](left,0) * (  1  -  NN->neuron_layers[L](left,0)  ) ); // 8:38 of Backpropagation Algorithm | Neural Networks video
            }
            all_neuron_deltas[L-1](left, 0) = delta; // populate the calculated deltas for the neurons of the left side layer into the corresponding all_neuron_deltas index
        }
    }

    // Back-propagates the weight gradients of all the weights in the network using the deltas calculated in the block before
    for (int L = DConDWs.size()-2 ; L>=0 ; L--){ // L goes 1, 0
        for (int left=0 ; left<NN->connection_layers[L].cols() ; left++){ // iterates through the left side neurons (which are arranged in the connections columns)
            for (int right=0 ; right<NN->connection_layers[L].rows() ; right++){ // iterates through the right side neurons (which are arranged in the connections rows)
                double delta_of_right_side_neuron = all_neuron_deltas[L](right, 0); // self explanitory
                double activation_of_left_side_neuron = NN->neuron_layers[L](left, 0); // self explanitory
                DConDWs[L](right, left) = delta_of_right_side_neuron * activation_of_left_side_neuron; // populates that entity, in the matrix, in the list, of DConDWs 
            }
        }
    }

    // Updates the the weights and biases with respect to their gradient relationship to the cost function and a learning rate (- ve to travel down the gradient)
    for (int L=0 ; L<NN->connection_layers.size() ; L++){ // L goes 0, 1, 2
        NN->connection_layers[L].array() += DConDWs[L].array() * learning_rate; // update the connection_layers by the negative gradient times a learning rate
        NN->bias_layers[L] += all_neuron_deltas[L] * learning_rate; // update the neuron biases (using deltas) by the negative gradient times a learning rate

        // cout << DConDWs.back() << endl << endl;

    }
}


void train_neural_network(Neural_Network* NN, const vector<vector<double>> training_data, const vector<vector<double>> labels, int epochs, double learning_rate){

    double lr = learning_rate;

    for (int i=0 ; i<epochs ; i++){

        Eigen::MatrixXd full_cost_matrix(training_data.size(), 1);

        for (int example=0 ; example<training_data.size() ; example++){

            Eigen::MatrixXd output_matrix = forward_pass(NN, training_data[example]); // forward pass works - i'll put my money on anyone's grandma
            Eigen::MatrixXd loss_matrix = calculate_loss_matrix(output_matrix, labels[example]); // (y - a^L)^2 // this is good fa sho
            full_cost_matrix(example,0) = loss_matrix.sum() / loss_matrix.rows(); // average the loss of all output predictions for this training example and append it to the full_cost_matrix matrix

            back_propagate(NN, loss_matrix, labels[example], lr);

            // cout << (double) example/training_data.size() * 100 / epochs << "% complete" << endl;

        }

        lr *= 0.7;

        cout << i << ":: " << full_cost_matrix.mean() << endl << endl;

    }
}

Eigen::MatrixXd predict_output_with_neural_network(Neural_Network* NN, vector<double> training_example){

    Eigen::MatrixXd output_matrix = forward_pass(NN, training_example);
    Eigen::MatrixXd output_matrix_normalized = (output_matrix.array() / output_matrix.sum()) * 100;

    Eigen::MatrixXd results(output_matrix.rows(), 2);

    for (int row=0 ; row < output_matrix.rows() ; row++){
        results(row, 0) = row;
        results(row, 1) = round(output_matrix_normalized(row,0) * 100.0) / 100.0;
    }
    
    return results ;
}


double calculate_network_accuracy(Neural_Network* NN, const vector<vector<double>> training_data, const vector<vector<double>> labels){

    int correct = 0;
    int incorrect = 0;

    for (int i=0 ; i<training_data.size() ; i++){

        Eigen::MatrixXd results = predict_output_with_neural_network(NN, training_data[i]);
        
        Eigen::MatrixXd::Index maxRow, maxCol;
        results.col(1).maxCoeff(&maxRow, &maxCol);

        // Find the iterator to the largest element in labels[i]
        auto max_it = std::max_element(labels[i].begin(), labels[i].end());

        // Calculate the index of the largest element
        int max_index = std::distance(labels[i].begin(), max_it);

        if (maxRow == max_index){
            correct+=1;
        }
        else{
            incorrect++;
        }
    }

    return (double)correct/(correct+incorrect) * 100;
}

void show_neural_network(Neural_Network* NN){
    Win32GUI gui;

    gui.drawNeuralNetwork(NN);
    // gui.redraw();
    gui.run();

}




#endif