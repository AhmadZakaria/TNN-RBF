#include <iostream>
#include "rbfn.cpp"
#include<math.h>
#include <string>
#include <iostream>
#include <fstream>
#include "Eigen/Dense"
namespace helper {


Eigen::MatrixXd readInputs(std::string filename, Eigen::MatrixXd &outputs) {
    std::fstream input_file(filename.c_str(), std::ios_base::in);
    if(!input_file.is_open()) {
        std::cerr << "Not open!! "<< std::endl;
        return Eigen::MatrixXd();
    }
    std::string s;
    int P, N, M;

    std::getline(input_file, s); // skip first line
    char hash;
    input_file >> hash; // skip hash character
    input_file >> s; // read P
    std::cout << "size of s: " << s.size() << " (" << s << ")" << " (" << hash << ")" << std::endl;
    P = std::stoi(s.substr(2), nullptr); // skip 'P=' and parse P
    input_file >> s; // read N
    N = std::stoi(s.substr(2), nullptr); // skip 'N=' and parse N
    input_file >> s; // read M
    M = std::stoi(s.substr(2), nullptr); // skip 'M=' and parse M

    Eigen::MatrixXd inputs;
    inputs.resize(P, N);
    //    delete outputs;
    outputs.resize(P, M);

    double a;
    for (int ex = 0; ex < P; ++ex) {
        for (int in = 0; in < N; ++in) {
            input_file >> a;
            //            std::cout<< " (" << P<<","<<N << ")" <<std::endl;
            (inputs)(ex,in) = a;
        }
        for (int teach = 0; teach < M; ++teach) {
            input_file >> a;
            //            std::cout<< " (" << P<<","<<teach << ") == " << a<<std::endl;
            (outputs)(ex, teach) = a;
        }
    }


    return inputs;
}
}
//using namespace Eigen;
//using namespace std;

int main() {
    srand(0);

    //  read inputs NxM (number of input and output neuron)
    //  set number of hidden layers
    //  set number of neurons per hidden layer
    //  set activations for every layer
    //  create the network
    //  forward propagate
    Eigen::MatrixXd teacher_output;
    Eigen::MatrixXd training_data = helper::readInputs("../training2_tr.dat", teacher_output);
    Eigen::MatrixXd test_output;
    Eigen::MatrixXd testing_data = helper::readInputs("../training2_ts.dat", test_output);

    RBFN *net = new RBFN(training_data,teacher_output,20);
    net->evaluate(testing_data,test_output);
    //    Eigen::MatrixXd* out ;//= net->forward_propagate();

    //    std::cout << *training_data << std::endl<< std::endl;
    //    std::cout << *teacher_output << std::endl<< std::endl;

    //    std::cout << *out << std::endl<< std::endl;

    //    double err = ((*teacher_output) - (*out)).squaredNorm();
    //    std::cout << "Sum of squared errors: "<< err <<std::endl;

}
