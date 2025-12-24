import os

import casadi as ca
import numpy as np


def staccatoe_fk():
  """
  Main function that sets up the robot kinematic model using CasADi.
  This function converts the MATLAB code with symbolic variables to Python with CasADi.

  Returns:
      tuple: (PHI_fun, J_fun) - functions to compute motor angles and Jacobian
  """
  # Define fixed parameters from the MATLAB code
  # Anti-parallelogram lengths
  r_ap = 55e-3
  l_apc = 265.9e-3
  l_su = 273.43966e-3
  r_sm1 = 66e-3

  # Lower spatial six-bar
  r_sm2 = 80e-3
  l_fbc = 95e-3
  l_sl = 83.45208e-3
  b_l = 22.5e-3
  b_r = -22.5e-3
  l_lhx = -55.60281e-3
  l_lhy = 30.5e-3
  l_lhz = -19.95864e-3
  l_rhx = -55.60281e-3
  l_rhy = -30.5e-3
  l_rhz = -19.95864e-3

  # Offset angles
  alpha_off1 = np.deg2rad(11.108078)
  gamma_1 = np.deg2rad(16.817357)
  gamma_2 = np.deg2rad(13.953575 - 0.65425044570618406947)
  gamma_3 = np.deg2rad(21.959907)

  # Middle five-bar and distal four-bar lengths
  l_c1 = 110e-3
  l_h1 = 73.28836505e-3
  l_h2 = 40e-3
  l_f1 = 33.59394724e-3
  l_f2 = 86.91072e-3
  l_h3 = 90e-3
  l_t1 = 26.92579661e-3

  # More offset angles
  gamma_4 = np.deg2rad(6.421644)
  gamma_5 = np.deg2rad(4.879181)
  gamma_6 = np.deg2rad(4.285071)
  gamma_7 = np.deg2rad(101.097381)
  gamma_8 = np.deg2rad(101.976037)
  alpha_off2 = np.deg2rad(-72.70180409)

  # Coordinate transform translations
  x_hu = 18.20808e-3
  y_hu = 0
  z_hu = -69.97354e-3
  x_23 = 33.47221e-3
  y_23 = 0
  z_23 = -2.85733e-3
  x_ta = 0
  y_ta = 0
  z_ta = l_sl

  # Gear ratios
  N_G1 = 6
  N_G2 = 9
  N_G3 = 9
  N_G4 = 9

  N_b = 40.0 / 9.0
  # 18 teeth pully
  # N_b_knee = 40.0 / 9.0
  # 21 teeth pully
  N_b_knee = (40.0 / 9.0) * (18.0 / 21.0)

  # Now let's define the symbolic joint variables
  q1 = ca.SX.sym("q1")  # hip roll
  q2 = ca.SX.sym("q2")  # hip pitch
  q3 = ca.SX.sym("q3")  # knee pitch
  q4 = ca.SX.sym("q4")  # ankle pitch
  q5 = ca.SX.sym("q5")  # ankle roll
  q6 = ca.SX.sym("q6")  # toe angle

  # Create rotation matrices
  R_yq4 = ca.vertcat(
    ca.horzcat(ca.cos(q4), 0, ca.sin(q4)),
    ca.horzcat(0, 1, 0),
    ca.horzcat(-ca.sin(q4), 0, ca.cos(q4)),
  )

  R_xq5 = ca.vertcat(
    ca.horzcat(1, 0, 0),
    ca.horzcat(0, ca.cos(q5), -ca.sin(q5)),
    ca.horzcat(0, ca.sin(q5), ca.cos(q5)),
  )

  #############################
  # ANKLE ACTUATION
  #############################

  # Lower four-bar (left)
  p_la1 = ca.vertcat(0, b_l, l_sl)
  p_lu3 = ca.vertcat(l_lhx, l_lhy, l_lhz)

  p_lu1 = R_yq4 @ R_xq5 @ p_lu3

  d_ly = ca.fabs(p_la1[1] - p_lu1[1])
  l_lxz = ca.sqrt(l_fbc**2 - d_ly**2)
  dX_l = p_la1[0] - p_lu1[0]
  dZ_l = p_la1[2] - p_lu1[2]
  dL_l = ca.sqrt(dX_l**2 + dZ_l**2)

  alpha_l = ca.asin(dX_l / ca.sqrt(dX_l**2 + dZ_l**2))
  beta_l = ca.acos((dL_l**2 + r_sm2**2 - l_lxz**2) / (2 * r_sm2 * dL_l))

  theta_sm2al = alpha_l + beta_l - (ca.pi / 2)
  theta_sm1al = theta_sm2al + alpha_off1 + gamma_2

  # Anti-parallelogram (left)
  k1 = l_su / r_sm1
  k2 = l_su / r_ap
  k3 = (l_apc**2 - (r_ap**2 + r_sm1**2 + l_su**2)) / (2 * r_ap * r_sm1)

  B1 = ca.sin(theta_sm1al) - k1
  B2 = ca.cos(theta_sm1al)
  B3 = k2 * ca.sin(theta_sm1al) + k3

  tan_q_2 = (-B1 - ca.sqrt(B1**2 + B2**2 - B3**2)) / (B2 + B3)
  q_ap_l_ = -ca.asin((2 * tan_q_2) / (1 + tan_q_2**2)) - gamma_1

  # Lower four-bar (right)
  p_ra1 = ca.vertcat(0, b_r, l_sl)
  p_ru3 = ca.vertcat(l_rhx, l_rhy, l_rhz)

  p_ru1 = R_yq4 @ R_xq5 @ p_ru3

  d_ry = ca.fabs(p_ra1[1] - p_ru1[1])
  l_rxz = ca.sqrt(l_fbc**2 - d_ry**2)
  dX_r = p_ra1[0] - p_ru1[0]
  dZ_r = p_ra1[2] - p_ru1[2]
  dL_r = ca.sqrt(dX_r**2 + dZ_r**2)

  alpha_r = ca.asin(dX_r / ca.sqrt(dX_r**2 + dZ_r**2))
  beta_r = ca.acos((dL_r**2 + r_sm2**2 - l_rxz**2) / (2 * r_sm2 * dL_r))

  theta_sm2ar = alpha_r + beta_r - (ca.pi / 2)
  theta_sm1ar = theta_sm2ar + alpha_off1 + gamma_2

  # Anti-parallelogram (right)
  B1 = ca.sin(theta_sm1ar) - k1
  B2 = ca.cos(theta_sm1ar)
  B3 = k2 * ca.sin(theta_sm1ar) + k3

  tan_q_2 = (-B1 - ca.sqrt(B1**2 + B2**2 - B3**2)) / (B2 + B3)
  q_ap_r_ = -ca.asin((2 * tan_q_2) / (1 + tan_q_2**2)) - gamma_1

  #############################
  # TOE ACTUATION
  #############################

  # Lower 4-bar
  n1 = l_t1 / l_f2
  n2 = l_t1 / l_h2
  n3 = (l_h2**2 + l_f2**2 + l_t1**2 - l_h3**2) / (2 * l_h2 * l_f2)

  C1 = ca.sin(gamma_6) + n1 * ca.sin(q6 + gamma_7)
  C2 = ca.cos(gamma_6) + n1 * ca.cos(q6 + gamma_7)
  C3 = n3 + n2 * ca.cos(q6 + gamma_7 - gamma_6)

  tan_q_2 = (C1 + ca.sqrt(C1**2 + C2**2 - C3**2)) / (C2 + C3)
  theta_5 = ca.acos((1 - tan_q_2**2) / (1 + tan_q_2**2))

  # Five-bar
  p_tuh = ca.vertcat(x_hu, y_hu, z_hu)
  R_ytheta5 = ca.vertcat(
    ca.horzcat(ca.cos(theta_5), 0, ca.sin(theta_5)),
    ca.horzcat(0, 1, 0),
    ca.horzcat(-ca.sin(theta_5), 0, ca.cos(theta_5)),
  )

  p_tuf2 = R_ytheta5 @ p_tuh
  p_tuf3 = p_tuf2 + ca.vertcat(x_23, y_23, z_23)

  p_tu1 = R_xq5 @ R_yq4 @ p_tuf3
  p_ta1 = ca.vertcat(x_ta, y_ta, z_ta)

  g1 = p_tu1[0]
  g2 = p_tu1[1]
  g3 = p_tu1[2]

  o1 = g1 / p_ta1[2]
  o2 = p_ta1[0] / p_ta1[2]
  o3 = g3 / p_ta1[2]

  D1 = 1 - o3
  D2 = o1 - o2
  D3 = (
    l_c1**2
    - (g2 - p_ta1[1]) ** 2
    - g1**2
    - g3**2
    - p_ta1[0] ** 2
    - p_ta1[2] ** 2
    - r_sm2**2
    + 2 * g1 * p_ta1[0]
    + 2 * g3 * p_ta1[2]
  ) / (2 * r_sm2 * p_ta1[2])

  tan_q_2 = (D1 - ca.sqrt(D1**2 + D2**2 - D3**2)) / (D2 + D3)
  theta_sm2t = ca.asin((2 * tan_q_2) / (1 + tan_q_2**2))

  # theta_sm2t = 2 * ca.atan(tan_q_2)

  # Anti-parallelogram
  theta_sm1t = theta_sm2t + alpha_off1 + gamma_2

  E1 = ca.sin(theta_sm1t) - k1
  E2 = ca.cos(theta_sm1t)
  E3 = k2 * ca.sin(theta_sm1t) + k3

  tan_q_2 = (-E1 - ca.sqrt(E1**2 + E2**2 - E3**2)) / (E2 + E3)
  q_tp_ = gamma_4 - ca.asin((2 * tan_q_2) / (1 + tan_q_2**2))
  # q_tp_ = gamma_4 - 2 * ca.atan(tan_q_2)

  # Final angle calculations
  q_ap_l = q_ap_l_ + q3  # Left ankle pulley angle w.r.t. thigh frame
  q_ap_r = q_ap_r_ + q3  # Right ankle pulley angle w.r.t. thigh frame
  q_tp = q_tp_ + q3  # Toe pulley angle w.r.t. thigh frame

  # Motor angles
  phi_thp = N_G4 * q2
  phi_thr = N_G4 * q1
  phi_knee = (N_G2 * N_b_knee) * q3
  phi_al = (N_G1 * N_b) * q_ap_l
  phi_ar = (N_G1 * N_b) * q_ap_r
  phi_toe = (N_G1 * N_b) * q_tp

  # Construct the vector of motor angles
  PHI = ca.vertcat(phi_thr, phi_thp, phi_knee, phi_al, phi_ar, phi_toe)

  # Construct the vector of joint angles
  q_vec = ca.vertcat(q1, q2, q3, q4, q5, q6)

  return ca.Function("fk", [q_vec], [PHI])


if __name__ == "__main__":
  fk_orig_func = staccatoe_fk()
  script_dir = os.path.dirname(os.path.abspath(__file__))
  fk_path = os.path.join(script_dir, "../casadi_functions/staccatoe_fk.casadi")

  fk_orig_func.save(fk_path)
