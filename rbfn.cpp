#include <cmath>
#include<iostream>
#include <fstream>


#include"Eigen/Dense"
#include"Eigen/StdVector"

#include "Kmeans/KMeansRexCoreInterface.h"

#define REAL double


using namespace Eigen;

class RBFN {
  private:
    int N;
    int K;
    int M;
    int P;
    REAL eta = 0.01;
    MatrixXd Xpn; // X(PxN): P Patterns, with N inputs each
    MatrixXd Ckn; // C(KxN): Center matrix. K vectors, N dimentional each
    MatrixXd Dpk; // D(PxK) Distance matrix for every pattern-neuron pair
    VectorXd Sk; // S(1xK) Widths of centers of neurons K
    MatrixXd Rpk; // R(PxK) RBF layer outputs: P outputs, of K length each.
    MatrixXd Wkm; // W(KxM) Weights between RBF and output
    MatrixXd Ypm; // Y(PxM) Network Outputs: P outputs, of M length each
    MatrixXd teacher_output; // (PxM) teacher's output for the input patterns


  public:
    RBFN(MatrixXd& training_inputs, MatrixXd& teacher_outputs,int k) {
        setInput(training_inputs);
        this->K = k;
        setTeacherOutput(teacher_outputs);
        init();

    }
    void initWidths(Eigen::ArrayXd z) {
        Eigen::ArrayXd c_size = Eigen::ArrayXd::Zero(K);
        Sk = Eigen::VectorXd::Zero(K);

        for (int i = 0; i < P; ++i) {
            int idx = z(i);
            c_size(idx)++;
            Sk(idx) += (Ckn.row(idx) - Xpn.row(i)).norm();
        }

        // normalize distance
        #pragma omp parallel for
        for (int i = 0; i < K; ++i) {
            Sk(i) /=  c_size(i);
        }

    }

    void initCenters() {
        Ckn.resize(K,N);
        Eigen::ArrayXd z = Eigen::ArrayXd::Zero(P);
        int n_iters = 200;
        RunKMeans(
            Xpn.data(),
            P, N, K,
            n_iters, 0, "plusplus",
            Ckn.data(),
            z.data()
        );
        IOFormat CleanFrmt(3, 0, " ", "\n", "[", "]");

        std::cout << K<<" - Centroids:" << Ckn.format(CleanFrmt) << std::endl;
        std::cout << z.rows()<<" - Zs:" << z.format(CleanFrmt) << std::endl;
        initWidths(z);
    }
    void initWeights() {
        Wkm = Eigen::MatrixXd::Random(K,M) * 0.5f;
    }

    void init() {
        initCenters();
        Rpk.resize(P,K);
        Dpk.resize(P,K);
        Ypm.resize(P,M);
        initWeights();

    }

    void calc_distances() {
        #pragma omp parallel for
        for (int p = 0; p < P; ++p) {
            MatrixXd xx = Xpn.row(p);
            for (int k = 0; k < K; ++k) {
                MatrixXd  cc = Ckn.row(k) ;
                MatrixXd dist = xx - cc;
                Dpk(p,k) = dist.norm();
            }
        }
    }

    void calc_rbf_outputs() {
        Rpk.resize(P,K);
        #pragma omp parallel for collapse(2)
        for (int p = 0; p < P; ++p)
            for (int k = 0; k < K; ++k) {
                Rpk(p,k) = gaussian(Dpk(p,k),Sk(k));
            }
    }

    void calc_network_outputs() {
        Ypm = Rpk * Wkm;
    }

    void forward_propagate() {
        calc_distances();
        calc_rbf_outputs();
        calc_network_outputs();
    }

    void backward_propagate() {
        MatrixXd temp  = ((Ypm)-(teacher_output)).transpose() / teacher_output.rows();
        MatrixXd delta_w = temp * Rpk;
        delta_w *= -eta;

        Wkm += delta_w.transpose();

    }

    void setInput(MatrixXd &in) {
        this->Xpn.resize(in.rows(),in.cols());
        this->Xpn            = in;
        this->N              = in.cols();
        this->P              = in.rows();

    }

    void setTeacherOutput(MatrixXd &tOut) {
        this->teacher_output.resize(tOut.rows(),tOut.cols());
        this->Ypm.resize(tOut.rows(),tOut.cols());
        this->teacher_output = tOut;
        this->M              = tOut.cols();

    }

    REAL evaluate(MatrixXd& test_input, MatrixXd& test_output, int iters = 5000) {
        std::fstream output_file("error.dat", std::ios_base::out);
        if(!output_file.is_open()) {
            std::cerr << "Not open!! "<< std::endl;
            return 0.0;
        }
        MatrixXd ins = Xpn, outs = teacher_output;
        for (int iter = 0; iter < iters; ++iter) { // repeat training
            setInput(ins);
            setTeacherOutput(outs);
            // batch train
            forward_propagate();
            backward_propagate();

            //calculate training error
            forward_propagate();
            REAL err = ((teacher_output) - (Ypm)).squaredNorm() / teacher_output.rows();
            std::cout << "[train] Sum of squared errors: "<< err <<std::endl;
            output_file <<iter<< "\t"<< err << "\t";

            //calculate testing error
            setInput(test_input);
            setTeacherOutput(test_output);
            forward_propagate();
            err = ((teacher_output) - (Ypm)).squaredNorm()/ teacher_output.rows();
            std::cout << "[test] Sum of squared errors: "<< err <<std::endl;
            output_file << err <<std::endl;


        }
        output_file.close();

    }

    ~RBFN() {
        Rpk.resize(0,0);
        Dpk.resize(0,0);
        Ckn.resize(0,0);
        Ypm.resize(0,0);
        Xpn.resize(0,0);
        Ypm.resize(0,0);
        Sk.resize(0);
    }

  private:
    static REAL gaussian(REAL dk, REAL sk) {
        // exp( -(d^2)/2*s^2 )
        REAL tmp = - (dk*dk) / (2 * sk*sk);
        REAL ret = std::exp(tmp);
        return ret;
    }
};
