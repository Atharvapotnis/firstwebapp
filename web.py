import streamlit as st
from bayes_opt import BayesianOptimization

# Function to optimize (Example)
def func(x):
    return -(x - 2) ** 2 + 5

# Streamlit UI
st.title("Bayesian Optimization Web App")
st.write("Enter value for x to optimize")

# User Input
x = st.number_input("Enter x:", min_value=-10.0, max_value=10.0, step=0.1)

# Bayesian Optimization
optimizer = BayesianOptimization(f=func, pbounds={'x': (-10, 10)}, random_state=1)
optimizer.maximize(init_points=2, n_iter=5)

st.write("Optimized Value:", optimizer.max['params'])
st.write("Max Function Value:", optimizer.max['target'])
