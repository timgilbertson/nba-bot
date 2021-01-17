"""
Script for evaluating Stock Trading Bot.

Usage:
  eval.py [--eval-stock=<eval_stock>] [--window-size=<window-size>] [--model-name=<model-name>] [--debug] [--max-spend=<max_spend]

Options:
  --window-size=<window-size>   Size of the n-day window stock data representation used as the feature vector. [default: 10]
  --model-name=<model-name>     Name of the pretrained model to use (will eval all models in `models/` if unspecified).
  --debug                       Specifies whether to use verbose logs during eval operation.
  --max-spend=<max_spend>       The amount of starting capital. [default: 5000]
"""

import os
import coloredlogs

from docopt import docopt


def main(eval_stock, model_name, debug, max_spend):
    """ Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python eval.py --help]
    """    
    _, test_data, _, _ = get_stock_data(eval_stock)
    date = test_data["game_date"].to_list()

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, pretrained=True, model_name=model_name)
        profit, _ = evaluate_model(agent, eval_data, window_size, debug, date, max_spend)
        show_eval_result(model_name, profit, initial_offset)
        
    # Multiple Model Evaluation
    else:
        for model in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model)):
                agent = Agent(window_size, pretrained=True, model_name=model)
                profit = evaluate_model(agent, eval_data, window_size, debug, max_spend)
                show_eval_result(model, profit, initial_offset)
                del agent


if __name__ == "__main__":
    args = docopt(__doc__)

    eval_stock = args["--eval-stock"]
    window_size = int(args["--window-size"])
    model_name = args["--model-name"]
    debug = args["--debug"]
    max_spend = int(args["--max-spend"])

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(eval_stock, window_size, model_name, debug, max_spend)
    except KeyboardInterrupt:
        print("Aborted")
