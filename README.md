# TNN-RBF
RBF network implementation as a part of Technical Neural Networks course at University of Bonn

Compilation:
cd build
cmake ..
make

To run the program, simply call the executable.


to plot the results using gnuplot

gnuplot
plot "error.dat" using 1:2 title "training error", "error.dat" using 1:3 title "testing error"
