language: PYTHON
name: "HPOlib.cv"

variable {
  name: "momentum"
  size: 1
  type: FLOAT
  min: 0.5
  max: 1
}
variable {
  name: "LOG_learning_rate1"
  size: 1
  type: FLOAT
  min: -9.9035
  max: -0.693147
}
variable {
  name: "LOG_learning_rate2"
  size: 1
  type: FLOAT
  min: -9.9035
  max: -0.693147
}
variable {
  name: "LOG_learning_rate3"
  size: 1
  type: FLOAT
  min: -9.9035
  max: -0.693147
}
variable {
  name: "LOG_weight_cost1"
  size: 1
  type: FLOAT
  min: -11.5129
  max: 0
}
variable {
  name: "LOG_weight_cost2"
  size: 1
  type: FLOAT
  min: -11.5129
  max: 0
}

variable {
  name: "LOG_weight_cost3"
  size: 1
  type: FLOAT
  min: -11.5129
  max: 0
}

variable {
  name: "dropout1"
  size: 1
  type: FLOAT
  min: 0.2
  max: 0.8
}

variable {
  name: "dropout2"
  size: 1
  type: FLOAT
  min: 0.2
  max: 0.8
}

variable {
  name: "lr_step"
  size: 1
  type: INT
  min: 1
  max: 4
}
variable {
  name: "LOG_w_init1"
  size: 1
  type: FLOAT
  min: -11.5129
  max: 0
}
variable {
  name: "LOG_w_init3"
  size: 1
  type: FLOAT
  min: -11.5129
  max: 0
}
variable {
  name: "LOG_w_init3"
  size: 1
  type: FLOAT
  min: -11.5129
  max: 0
}
