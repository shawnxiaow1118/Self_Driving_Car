#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 12;
double dt = 0.08;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;
const double ref_v = 80.0;
const double Ccte = 1200.0;
const double Cepsi = 1200.0;
const double Cv = 1.0;
const double Cdelta = 100.0;
const double Ca = 10.0;
const double Cddelta = 600.0;
const double Cda = 10.0;

size_t x_start = 0;
size_t y_start = x_start+N;
size_t psi_start = y_start+N;
size_t v_start = psi_start+N;
size_t cte_start = v_start+N;
size_t epsi_start = cte_start+N;
size_t delta_start = epsi_start+N;
size_t a_start = delta_start+N-1;



// help function to evaluate the value of Polynomial function given coeffeicients
AD<double> polyeval(Eigen::VectorXd coeffs, AD<double> x) {
  AD<double> result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    AD<double> coef = coeffs[i];
    result += coef * CppAD::pow(x, i);
  }
  return result;
}



// help function to evaluate the value of derivative of a Polynomial function given coeffeicients
AD<double> dpolyeval(Eigen::VectorXd coeffs, AD<double> x) {
  AD<double> result = 0.0;
  for (int i = 1; i < coeffs.size(); i++) {
    AD<double> index = i;
    AD<double> coef = coeffs[i];
    result += index*coef * CppAD::pow(x, i-1);
  }
  return result;
}



class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.
    // fg[0] is the objective of the optimization problem
    fg[0] = 0;
    // penalize on error compared with reference states
    for(int t = 0; t < N; t++) {
      fg[0] += Ccte*CppAD::pow(vars[cte_start+t], 2);
      fg[0] += Cepsi*CppAD::pow(vars[epsi_start+t], 2);
      fg[0] += Cv*CppAD::pow(vars[v_start+t]-ref_v, 2);
    }

    // penalize on actuator, want minimize the actuator put onto the
    for(int t = 0; t < N-1; t++) {
      fg[0] += Cdelta*CppAD::pow(vars[delta_start+t], 2);
      fg[0] += Ca*CppAD::pow(vars[a_start+t], 2);
    }

    // penalize the change between consecutive actuators to prevent erratic change
    for(int t = 0; t < N-2; t++) {
      fg[0] += Cddelta*CppAD::pow(vars[delta_start+t+1]-vars[delta_start+t],2);
      fg[0] += Cda*CppAD::pow(vars[a_start+t+1]-vars[a_start+t],2);
    }

    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints, based on simple vehicle model
    for (int i = 0; i < N - 1; i++) {

      AD<double> x_cur = vars[x_start+i];
      AD<double> y_cur = vars[y_start+i];
      AD<double> psi_cur = vars[psi_start+i];
      AD<double> v_cur = vars[v_start+i];
      AD<double> cte_cur = vars[cte_start+i];
      AD<double> epsi_cur = vars[epsi_start+i];


      AD<double> x_next = vars[x_start+i+1];
      AD<double> y_next = vars[y_start+i+1];
      AD<double> psi_next = vars[psi_start+i+1];
      AD<double> v_next = vars[v_start+i+1];
      AD<double> cte_next = vars[cte_start+i+1];
      AD<double> epsi_next = vars[epsi_start+i+1];


      AD<double> delta_cur = vars[delta_start+i];
      AD<double> a_cur = vars[a_start+i];

      AD<double> ydes_cur = polyeval(coeffs, x_cur);
      AD<double> psides_cur = CppAD::atan(dpolyeval(coeffs, x_cur));

      fg[x_start+i+2] = x_next-(x_cur+v_cur*CppAD::cos(psi_cur)*dt);
      fg[y_start+i+2] = y_next-(y_cur+v_cur*CppAD::sin(psi_cur)*dt);
      fg[psi_start+i+2] = psi_next-(psi_cur-v_cur*delta_cur/Lf*dt);
      fg[v_start+i+2] = v_next-(v_cur+a_cur*dt);
      fg[cte_start+i+2] = cte_next-(ydes_cur-y_cur+v_cur*CppAD::sin(epsi_cur)*dt);
      fg[epsi_start+i+2] = epsi_next-(psi_cur-psides_cur-v_cur/Lf*delta_cur*dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;


  // 4 * 10 + 2 * 9
  size_t n_vars = 6*N + 2*(N-1);
  // TODO: Set the number of constraints
  size_t n_constraints = 6*N;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // TODO: Set lower and upper limits for variables.

  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start] = state[0];
  constraints_lowerbound[y_start] = state[1];
  constraints_lowerbound[psi_start] = state[2];
  constraints_lowerbound[v_start] = state[3];
  constraints_lowerbound[cte_start] = state[4];
  constraints_lowerbound[epsi_start] = state[5];

  constraints_upperbound[x_start] = state[0];
  constraints_upperbound[y_start] = state[1];
  constraints_upperbound[psi_start] = state[2];
  constraints_upperbound[v_start] = state[3];
  constraints_upperbound[cte_start] = state[4];
  constraints_upperbound[epsi_start] = state[5];

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  vector<double> result;

  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);


  for(int i = 0 ; i < N-1; i++) {
    result.push_back(solution.x[x_start+i+1]);
    result.push_back(solution.x[y_start+i+1]);
  }

    return result;
}
