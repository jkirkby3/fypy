import os
from scipy.io import loadmat as loadmat
import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import abstractmethod

import unittest
from tqdm import tqdm
from itertools import product

from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
from fypy.model.sv.HestonDEJumps import HestonDEJumps


@dataclass
class MarketParams:
    S0: float = 100
    r: float = 0.05
    q: float = 0


class PricerParams:
    def __init__(self, use_P: bool = False):
        if use_P:
            self.P: int = 5
            self.Pbar: int = 3
        else:
            self.N = 2**10


class Curves:
    def __init__(self, mkt: MarketParams):
        self.disc_curve = DiscountCurve_ConstRate(rate=mkt.r)
        self.div_disc = DiscountCurve_ConstRate(rate=mkt.q)
        self.fwd = EquityForward(
            S0=mkt.S0, discount=self.disc_curve, divDiscount=self.div_disc
        )


class Matlab:
    def __init__(self, option_name: str, barrier: bool = False):
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        file_path: str = os.path.join(
            self.script_dir,
            "tests",
            "numerical_values_for_testing",
            f"{option_name}StoPrices.mat",
        )
        file = loadmat(file_path)
        self.prices = file["prices"]
        self.W = file["lin_W"].flatten()
        self.T = file["lin_T"].flatten()
        self.M = file["lin_M"].flatten()
        if barrier:
            self.H = file["lin_H"].flatten()
        self.params = file["params"]


class Printer:
    def __init__(self):
        self._colors_code = {
            "PURPLE": "\033[95m",
            "CYAN": "\033[96m",
            "DARKCYAN": "\033[36m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "YELLOW": "\033[93m",
            "RED": "\033[91m",
            "BOLD": "\033[1m",
            "UNDERLINE": "\033[4m",
            "END": "\033[0m",
            "ITALIC": "\u001b[3m",
        }

    def write(
        self, text: str, color: str = "END", style: list[str] = [], indent: int = 0
    ):
        style_code = "".join([self._colors_code[st] for st in style])
        print(
            self._colors_code[color]
            + style_code
            + indent * "\t"
            + text
            + self._colors_code["END"]
        )

    def write_begin(self, option_name: str):
        color = "DARKCYAN"
        msg = " Test " + option_name + " Stochastic "
        hashtag_number = 10
        len_hashtag = len(msg) + 2 * hashtag_number
        self.write("\n\n" + len_hashtag * "#", color=color)
        self.write(hashtag_number * "#" + msg + hashtag_number * "#", color=color)
        self.write(len_hashtag * "#" + "\n", color=color)

    def write_parameters(
        self, idx_param: int, number_param_set: int, params: np.ndarray
    ):
        self.write(
            f"(*) Parameters set n°{idx_param}/{number_param_set-1}:     {list(params)}\n",
            indent=1,
            style=["BOLD"],
        )

    def write_success(self):
        self.write(
            "-> Parameters set successfully passed the test. \n",
            color="GREEN",
            style=["ITALIC"],
            indent=4,
        )

    def write_conclusion(self):
        self.write(
            "Recapitulative DataFrame: \n",
            style=["BOLD", "UNDERLINE"],
            color="DARKCYAN",
        )


class ErrorLog:
    def __init__(self, number_param_set: int):
        self.number_param_set = number_param_set
        self.errors = {i: [] for i in range(number_param_set)}
        self.nan_number = {i: 0 for i in range(number_param_set)}
        self.option_number = {i: 0 for i in range(number_param_set)}

    def add_error(self, param_set_idx: int, error: float):
        self.errors[param_set_idx].append(error)

    def add_nan(self, param_set_idx: int):
        self.nan_number[param_set_idx] += 1

    def get_error_max_idx(self, param_set_idx: int):
        return np.max(np.abs(self.errors[param_set_idx]))

    def get_error_list(self):
        error_list = []
        for i in self.errors:
            error_list.append(self.get_error_max_idx(i))
        return error_list

    def build_recap_df(self):
        error_list = self.get_error_list()
        data = {
            "Parameters set index": list(range(self.number_param_set)),
            "Max Error": error_list,
            "NaN Number": list(self.nan_number.values()),
            "Option Number": list(self.option_number.values()),
        }
        df = pd.DataFrame.from_dict(data)
        return df


class GenericTest:

    def __init__(self, option_name: str):
        self.option_name = option_name

    def test_psv(self):
        self._initialization()
        for idx_param in range(len(self.matlab.params)):
            self._update_pricer(self.matlab.params, idx_param)
            self._count_option(idx_param)
            self._test_idx_param(idx_param)
            self.printer.write_success()
        self.display_conclusion()

    def _test_idx_param(self, idx_param):

        for tuple_index in tqdm(self.product):
            list_index = list(tuple_index)
            price = self._try_price(idx_param, list_index)
            self._check_error_and_type(idx_param, list_index, price)

    @abstractmethod
    def _set_product(self):
        raise NotImplementedError

    @abstractmethod
    def _set_option_constants(self):
        raise NotImplementedError

    @abstractmethod
    def _get_price(self):
        raise NotImplementedError

    @abstractmethod
    def _set_pricer(self):
        raise NotImplementedError

    def _count_option(self, idx_param: int):
        self.error_log.option_number[idx_param] = np.prod(
            self.matlab.prices[idx_param].shape
        )

    def _try_price(self, idx_param, list_index):
        try:
            price = self._get_price(list_index)
        except:
            price = np.nan
            self.error_log.add_nan(idx_param)
        return price

    def _set_model(self):
        self.model = HestonDEJumps(self.curves.fwd, self.curves.disc_curve)

    # display the log error/NaN
    def display_conclusion(self):
        self.printer.write_conclusion()
        df = self.error_log.build_recap_df()
        print(df)

    # update the parameters of the pricer
    def _update_pricer(self, params: np.ndarray, idx_param: int):
        param = params[idx_param]
        self.printer.write_parameters(idx_param, len(params), param)
        self.model.set_params(param)
        self._set_pricer()
        return

    def _initialization(self):
        use_P = self.option_name == "Asian"
        is_barrier = self.option_name == "Barrier"

        self.printer = Printer()
        self.printer.write_begin(self.option_name)
        self.mkt = MarketParams()
        self.pricer_params = PricerParams(use_P=use_P)
        self.curves = Curves(self.mkt)
        self.matlab = Matlab(self.option_name, is_barrier)
        self._set_product()
        self._set_option_constants()
        self.error_log = ErrorLog(len(self.matlab.params))
        self._set_model()

    def _try_price(self, idx_param, list_index):
        try:
            price = self._get_price(list_index)
        except:
            price = np.nan
            self.error_log.add_nan(idx_param)
        return price

    def _check_error_and_type(self, idx_param, list_index, price):
        index = tuple([idx_param] + list_index)
        if np.isnan(price) != np.isnan(self.matlab.prices[index]):
            raise ValueError("Python output type should be consistent with Matlab.")
        if not np.isnan(price):
            self.check_equality(price, self.matlab.prices[index])
            error = price - self.matlab.prices[index]
            self.error_log.add_error(idx_param, error)

    def _set_product(self):
        match self.option_name:
            case "Barrier":
                self.product = list(
                    product(
                        range(len(self.matlab.W)),
                        range(len(self.matlab.T)),
                        range(len(self.matlab.M)),
                        range(len(self.matlab.H)),
                    )
                )
            case _:
                self.product = list(
                    product(
                        range(len(self.matlab.W)),
                        range(len(self.matlab.T)),
                        range(len(self.matlab.M)),
                    )
                )

    @abstractmethod
    def check_equality(self, price: float, matlab_price: float):
        raise NotImplementedError
