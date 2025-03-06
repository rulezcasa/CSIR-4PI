#!/bin/bash

# Set environment name
ENV_NAME="ATDQN_harish"

conda create --name $ENV_NAME -y

run_in_tmux() {
    SESSION_NAME=$1
    SCRIPT_NAME=$2

    tmux kill-session -t $SESSION_NAME 2>/dev/null

    tmux new-session -d -s $SESSION_NAME "export WANDB_API_KEY=bb9c65d370b2dd4ecf18c352a8e6b0a8ec928a22 && conda activate $ENV_NAME && python $SCRIPT_NAME"

    echo "Started $SCRIPT_NAME in tmux session '$SESSION_NAME' and detached."
}

conda activate $ENV_NAME  
pip install -r requirements.txt

echo "Conda environment '$ENV_NAME' is ready with dependencies installed."

FILES=("pong_ATDQN.py" "enduro_ATDQN.py" "breakout_ATDQN.py" "seaquest_ATDQN.py")

for i in "${!FILES[@]}"; do
    run_in_tmux "session_$i" "${FILES[$i]}"
done

echo "All scripts are running in separate tmux sessions and detached."
