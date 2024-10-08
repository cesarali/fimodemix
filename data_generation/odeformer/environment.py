# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

# Add the current directory to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import re
import traceback
from collections import defaultdict
from logging import getLogger
from typing import Union
import torch
# import math
import numpy as np

import generators
import simplifiers

from .utils import *


TIMEOUT = 2

SPECIAL_WORDS = [
    "<EOS>",
    "<X>",
    "</X>",
    "<Y>",
    "</Y>",
    "</POINTS>",
    "<INPUT_PAD>",
    "<OUTPUT_PAD>",
    "<MASK>",
    "<PAD>",
    "(",
    ")",
    "SPECIAL",
    "OOD_unary_op",
    "OOD_binary_op",
    "OOD_constant",
]
logger = getLogger()
regex = r"^-?\d+\.\d{4}( \+ -?\d+\.\d{4})? \* x_\d+$"
SKIP_ITEM = "SKIP_ITEM"


class FunctionEnvironment(object):
    TRAINING_TASKS = {"functions"}

    def __init__(self, params):
        self.params = params
        self.rng = None
        self.float_precision = params.float_precision
        self.max_size = None

        self.generator = generators.RandomFunctions(params, SPECIAL_WORDS)
        self.float_encoder = self.generator.float_encoder
        self.float_words = self.generator.float_words
        self.equation_encoder = self.generator.equation_encoder
        self.equation_words = self.generator.equation_words
        if params.use_two_hot:
            self.constant_words = self.equation_encoder.constant_encoder.symbols
        else:
            self.equation_words += self.float_words
            self.constant_words = None
        self.simplifier = simplifiers.Simplifier(self.generator)

        # number of words / indices
        self.float_id2word = {i: s for i, s in enumerate(self.float_words)}
        self.equation_id2word = {i: s for i, s in enumerate(self.equation_words)}
        self.float_word2id = {s: i for i, s in self.float_id2word.items()}
        self.equation_word2id = {s: i for i, s in self.equation_id2word.items()}
        if params.use_two_hot:
            _offset = max(self.equation_word2id.values()) + 1
            self.constant_id2word = {i + _offset: s for i, s in enumerate(self.constant_words)}
            self.constant_word2id = {s: i for i, s in self.constant_id2word.items()}

        for ood_unary_op in self.generator.extra_unary_operators:
            self.equation_word2id[ood_unary_op] = self.equation_word2id["OOD_unary_op"]
        for ood_binary_op in self.generator.extra_binary_operators:
            self.equation_word2id[ood_binary_op] = self.equation_word2id["OOD_binary_op"]
        if self.generator.extra_constants is not None:
            for c in self.generator.extra_constants:
                self.equation_word2id[c] = self.equation_word2id["OOD_constant"]

        assert len(self.float_words) == len(set(self.float_words))
        assert len(self.equation_word2id) == len(set(self.equation_word2id))
        self.n_words = params.n_words = len(self.equation_words)
        logger.info(f"vocabulary: {len(self.float_word2id)} float words, {len(self.equation_word2id)} equation words")

        if params.debug:
            global TIMEOUT
            TIMEOUT = 100000

    def mask_from_seperator(self, x, sep):
        sep_id = self.float_word2id[sep]
        alen = torch.arange(x.shape[0], dtype=torch.long, device=x.device).unsqueeze(-1).repeat(1, x.shape[1])
        sep_id_occurence = torch.tensor(
            [torch.where(x[:, i] == sep_id)[0][0].item() if len(torch.where(x[:, i] == sep_id)[0]) > 0 else -1 for i in range(x.shape[1])]
        )
        mask = alen > sep_id_occurence
        return mask

    def batch_equations(self, equations):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """

        lengths = torch.LongTensor([2 + len(eq) for eq in equations])
        if self.equation_encoder.constant_encoder is not None:
            sent = torch.DoubleTensor(lengths.max().item(), lengths.size(0)).fill_(self.float_word2id["<PAD>"])
        else:
            sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.float_word2id["<PAD>"])
        sent[0] = self.equation_word2id["<EOS>"]
        for i, eq in enumerate(equations):
            sent[1 : lengths[i] - 1, i].copy_(eq)
            sent[lengths[i] - 1, i] = self.equation_word2id["<EOS>"]
        return sent, lengths

    def word_to_idx(self, words, float_input=True):
        """
        Maps from actual words (e.g. 'add' or '1.43') to ids. Ids can be floats if we use two-hot encoding.
        `words`: List[List[str]], e.g. from samples["tree_encoded"]
        """

        if float_input:
            return [[torch.LongTensor([self.float_word2id[dim] for dim in point]) for point in seq] for seq in words]
        elif self.equation_encoder.constant_encoder is not None:
            id_offset = max(self.equation_word2id.values())
            output = []
            for eq in words:
                lst = []
                for w in eq:
                    try:
                        # NOTE: Assuming that encoder embedding layer and decoder embedding are NOT shared
                        # otherwise, we need to ensure that indices of float encoder and constant encoder do not overlap
                        lst.append(float(w) + id_offset - self.equation_encoder.constant_encoder.min)
                        # in the line above, float(w) raises an error for non-constant words
                    except ValueError:
                        lst.append(self.equation_word2id[w])

                output.append(lst)
            return output
        else:
            return [[self.equation_word2id[w] for w in eq] for eq in words]

    # def ids_to_two_hot(self, ids, support_size: int):
    #     """
    #     Take a list of int or float 'ids' and convert them to 2-hot representation.
    #     For int ids, this is equivalent to 1-hot representation.
    #     """
    #     id_left = ids.floor()
    #     id_right = id_left + 1
    #     prob_left = 1 - (ids - id_left)
    #     prob_right = 1 - prob_left
    #     logits = torch.zeros(ids.shape[0], ids.shape[1], support_size).type_as(ids)
    #     return (
    #         logits.scatter_(dim=2, index=id_left.long().unsqueeze(-1), src=prob_left.unsqueeze(-1))
    #         .scatter_(dim=2, index=id_right.long().unsqueeze(-1), src=prob_right.unsqueeze(-1))
    #         .squeeze(1)
    #     )

    # def topk_decode_two_hot(
    #     self, logits: torch.Tensor, topk_idx: torch.Tensor, unfinished_sents: Union[None, torch.Tensor], apply_softmax: bool = True
    # ):
    #     """
    #     Take logits and topk indices, whenever a topk index refers to the indices that represent two-hot encoded
    #     constants, decode the selected constant. The decoding process works as follows:
    #     1. Apply softmax across all logits that represent constants.
    #     2. Among the two top-k neighboring indices, select the one whose corresponding probability is larger.
    #     3. Decode as topk * prob_{topk} + neighbor * prob_{neighbor}.
    #     This function is necessary as topk returns a single index but we need two indices to represent two-hot encoded
    #     constants. This problem is solved by allowing for "float" ids, e.g. we combine the two indices that represent a
    #     constant into a single float. The float is still in "id-space", i.e., the returned float ids do not directly
    #     correspond to the encoded float but to the float plus an offset that accounts for the remaining non-two-hot
    #     tokens (e.g. tokens for operators and variables).

    #     `logits`:
    #         Tensor, (bs, vocab size). Batch of logits.
    #     `topk_idx`:
    #         Tensor, (bs,). Batch of selected indices for decoding.
    #     `unfinished_sents`:
    #         Tensor, (bs, ). Batch of indices that indicate batch elements that are part of a not-yet-finished sequence.
    #         If a sequence is already finished, we do not have to do any constant decoding, so this is an opportunity for
    #         a short-cut.

    #     Returns:
    #     `topk_idx`:
    #         Tensor, (bs,).
    #         The original tensor with topk indices that correspond to two-hot constants converted to float indices.
    #     `constants_mask`:
    #         Tensor, (bs,).
    #         A boolean tensor of same shape as `top_idx` that indicates if the corresponding element in `top_idx` is a
    #         decoded constant.

    #     Caution: topk_idx.dtype changes in place from int64 to torch.float32 or torch.double.
    #     """

    #     id_offset = max(self.equation_word2id.values()) + 1
    #     constants_mask = topk_idx.squeeze() >= id_offset
    #     if constants_mask is not None:
    #         constants_mask = constants_mask.to(torch.int) * unfinished_sents
    #         constants_mask = constants_mask.to(torch.bool)
    #     if not constants_mask.any():
    #         # print(f"NO : topk_decode_two_hot: topk_ids = {topk_idx}")
    #         return topk_idx, constants_mask
    #     # print(f"YES: topk_decode_two_hot: topk_ids = {topk_idx}")
    #     if apply_softmax:
    #         probs = torch.nn.functional.softmax(logits[constants_mask, id_offset:], dim=1)
    #     else:
    #         probs = logits[constants_mask, :]

    #     neg_infs = torch.ones((probs.shape[0], 1), device=logits.device).fill_(-torch.inf)
    #     probs = torch.cat([neg_infs, probs, neg_infs], dim=1)
    #     topk_idx[constants_mask] += 1  # compensate for the prepended -inf
    #     # indices
    #     left_neighbor = (topk_idx[constants_mask] - 1).reshape(-1, 1) - id_offset
    #     selected = (topk_idx[constants_mask]).reshape(-1, 1) - id_offset
    #     right_neighbor = (topk_idx[constants_mask] + 1).reshape(-1, 1) - id_offset
    #     # probs
    #     left_prob = torch.gather(input=probs, dim=1, index=left_neighbor)
    #     selected_prob = torch.gather(input=probs, dim=1, index=selected)
    #     right_prob = torch.gather(input=probs, dim=1, index=right_neighbor)
    #     # choice
    #     take_left = left_prob > right_prob
    #     take_right = ~take_left
    #     best_neighbor = left_neighbor * take_left + right_neighbor * take_right
    #     neighbor_prob = torch.gather(input=probs, dim=1, index=best_neighbor)
    #     # value
    #     selected_value = selected + self.equation_encoder.constant_encoder.min
    #     neighbor_value = best_neighbor + self.equation_encoder.constant_encoder.min
    #     # un-compensate for the prepended -inf
    #     selected_value -= 1
    #     neighbor_value -= 1
    #     value = torch.squeeze(selected_value * selected_prob + neighbor_value * neighbor_prob).to(torch.double)
    #     id_value = value + id_offset - self.equation_encoder.constant_encoder.min

    #     # print("selected", selected)
    #     # print("selected_value", selected_value)
    #     # print("selected_prob", selected_prob)

    #     # print("best_neighbor", best_neighbor)
    #     # print("neighbor_value", neighbor_value)
    #     # print("neighbor_prob", neighbor_prob)

    #     # print("id_offset", id_offset)
    #     # print("self.equation_encoder.constant_encoder.min", self.equation_encoder.constant_encoder.min)

    #     # import matplotlib.pyplot as plt
    #     # try:
    #     #     plt.plot(probs.T, ".")
    #     # except TypeError:
    #     #     plt.plot(probs.cpu().numpy().T, ".")
    #     # plt.title(value.detach().cpu().numpy().round(3))
    #     # plt.show()

    #     topk_idx = topk_idx.to(id_value.dtype)
    #     topk_idx[constants_mask] = id_value

    #     # print(f"topk_decode_two_hot: topk_ids = {topk_idx}")
    #     return topk_idx, constants_mask

    def word_to_infix(self, words, is_float=True, str_array=True):
        if is_float:
            m = self.float_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            m = self.equation_encoder.decode(words)
            if m is None:
                # print("word_to_infix() is None", words, "\n")
                return None
            if str_array:
                return m.infix()
            else:
                return m

    def wrap_equation_floats(self, tree, constants):
        prefix = tree.prefix().split(",")
        j = 0
        for i, elem in enumerate(prefix):
            if elem.startswith("CONSTANT"):
                prefix[i] = str(constants[j])
                j += 1
        assert j == len(constants), "all constants were not fitted"
        assert "CONSTANT" not in prefix, "tree {} got constant after wrapper {}".format(tree, constants)
        tree_with_constants = self.word_to_infix(prefix, is_float=False, str_array=False)
        return tree_with_constants

    def idx_to_infix(
        self,
        lst,
        is_float=True,
        str_array=True,
        is_two_hot=None,
    ):
        # print("idx_to_infix", lst)
        if is_float:
            idx_to_words = [self.float_id2word[int(i)] for i in lst]
        else:
            id_offset = max(self.equation_word2id.values())
            idx_to_words = []
            for term_i, term in enumerate(lst):
                if self.params.use_two_hot:
                    if is_two_hot[term_i]:
                        # two-hot constants: we need to undo the offset introduced in word_to_idx()
                        idx_to_words.append(f"{term - id_offset + self.equation_encoder.constant_encoder.min:+}")
                else:
                    idx_to_words.append(self.equation_id2word[int(term)])

        return self.word_to_infix(idx_to_words, is_float, str_array)

    def gen_expr(self, train, nb_binary_ops=None, nb_unary_ops=None, dimension=None, n_points=None):
        errors = defaultdict(int)
        if not train or self.params.use_controller:
            # if nb_unary_ops is None:
            #     nb_unary_ops = self.rng.randint(
            #         self.params.min_unary_ops, self.params.max_unary_ops + 1
            #     )
            if dimension is None:
                dimension = self.rng.randint(self.params.min_dimension, self.params.max_dimension + 1)
        while True:
            try:
                expr, error = self._gen_expr(
                    train, nb_binary_ops=nb_binary_ops, nb_unary_ops=nb_unary_ops, dimension=dimension, n_points=n_points
                )
                if error:
                    if self.params.debug:
                        print(error)
                    errors[error[0]] += 1
                    assert False
                if self.params.skip_linear and bool(re.match(regex, str(expr["tree"]))):
                    continue
                return expr, errors
            except (AssertionError, MyTimeoutError) as e:
                if self.params.debug:
                    print(f"Caught {type(e).__name__}: {str(e)}")
                continue  # Loop restarts
            except:
                if self.params.debug:
                    print(traceback.format_exc())
                continue

    def _gen_expr(self, train, nb_binary_ops=None, nb_unary_ops=None, dimension=None, n_points=None):
        (
            tree,
            dimension,
            nb_unary_ops,
            nb_binary_ops,
        ) = self.generator.generate_multi_dimensional_tree(
            rng=self.rng, nb_unary_ops=nb_unary_ops, nb_binary_ops=nb_binary_ops, dimension=dimension
        )

        if tree is None:
            return {"tree": tree}, ["bad tree"]

        # effective_dimension = self.generator.relabel_variables(tree)
        # if dimension == 0 or (
        #     self.params.enforce_dim and effective_dimension < dimension
        # ):
        #     return {"tree": tree}, ["bad input dimension"]

        for op in self.params.operators_to_not_repeat.split(","):
            if op and tree.prefix().count(op) > 1:
                return {"tree": tree}, ["ops repeated"]

        if self.params.use_sympy:
            len_before = len(tree.prefix().split(","))
            try:
                tree = (
                    self.simplifier.simplify_tree(tree, expand=self.params.expand, resimplify=self.params.simplify)
                    if self.params.use_sympy
                    else tree
                )
            except:
                return {"tree": tree}, ["simplification error"]
            len_after = len(tree.prefix().split(","))
            if tree is None or len_after > 2 * len_before:
                return {"tree": tree}, ["simplification error"]

        if n_points is None:
            n_points = self.rng.randint(min(self.params.min_points, self.params.max_points), self.params.max_points + 1)

        times, trajectory = None, None
        if self.params.solve_ode:
            # generate trajectory
            tree, datapoints = self.generator.generate_datapoints(tree=tree, rng=self.rng, dimension=dimension, n_points=n_points)
            if datapoints is None:
                return {"tree": tree}, ["datapoint generation error for tree {}".format(tree)]

            times, trajectory = datapoints
            n_points = trajectory.shape[0]

        # # encode tree
        # tree_encoded = self.equation_encoder.encode(tree)
        # skeleton_tree, _ = self.generator.function_to_skeleton(tree)
        # skeleton_tree_encoded = self.equation_encoder.encode(skeleton_tree)

        # if self.equation_encoder.constant_encoder is not None:

        #     def is_number(s):
        #         try:
        #             float(s)
        #             return True
        #         except ValueError:
        #             return False

        #     assert all([is_number(x) or x in self.equation_word2id for x in tree_encoded]), "bad tokens in encoded tree:".format()
        # else:
        #     assert all([x in self.equation_word2id for x in tree_encoded]), "tree_encoded: {}\n:".format(
        #         [(token, token in self.equation_word2id) for token in tree_encoded]
        #     )

        info = {
            "n_points": n_points,
            "n_unary_ops": sum(nb_unary_ops),
            "n_binary_ops": sum(nb_binary_ops),
            "dimension": dimension,
        }

        expr = {
            "times": times,
            "trajectory": trajectory,
            # "tree_encoded": tree_encoded,
            # "skeleton_tree_encoded": skeleton_tree_encoded,
            "tree": tree,
            # "skeleton_tree": skeleton_tree,
            "infos": info,
        }

        return expr, []

    def _create_noise(
        self,
        trajectory: np.ndarray,
        train: Union[bool, None] = None,
        gamma: Union[None, float] = None,
        seed: Union[int, None] = None,
    ):
        """Returns noise"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            try:
                rng = self.env.rng
            except:
                rng = np.random.RandomState(0)
        if gamma is None:
            gamma = rng.choice(np.linspace(0, self.params.train_noise_gamma, 5)) if train else self.params.eval_noise_gamma
        return gamma * trajectory * np.random.randn(*trajectory.shape), gamma

    def _subsample_trajectory(
        self,
        times: np.ndarray,
        trajectory: np.ndarray,
        train: Union[bool, None] = None,
        subsample_ratio: Union[None, float] = None,
        seed: Union[None, int] = None,
    ):
        """Applies subsampling in-place."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            try:
                rng = self.env.rng
            except:
                rng = np.random.RandomState(0)
        if subsample_ratio is None:
            subsample_ratio = (
                rng.choice(np.linspace(0, self.params.train_subsample_ratio, 5)) if train else self.params.eval_subsample_ratio
            )
        indices_to_remove = rng.choice(
            trajectory.shape[0],
            int(trajectory.shape[0] * subsample_ratio),
            replace=False,
        )
        trajectory = np.delete(trajectory, indices_to_remove, axis=0)
        times = np.delete(times, indices_to_remove, axis=0)
        return times, trajectory, subsample_ratio

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--use_queue",
            type=bool_flag,
            default=True,
            help="whether to use queue",
        )

        parser.add_argument("--collate_queue_size", type=int, default=2000)

        parser.add_argument(
            "--use_sympy",
            type=bool_flag,
            default=True,
            help="Whether to use sympy parsing (basic simplification)",
        )
        parser.add_argument(
            "--expand",
            type=bool_flag,
            default=False,
            help="Whether to use sympy expansion",
        )
        parser.add_argument(
            "--simplify",
            type=bool_flag,
            default=False,
            help="Whether to use further sympy simplification",
        )
        parser.add_argument(
            "--use_abs",
            type=bool_flag,
            default=False,
            help="Whether to replace log and sqrt by log(abs) and sqrt(abs)",
        )

        # encoding
        parser.add_argument(
            "--operators_to_use",
            type=str,
            default="sin:1,inv:1,pow2:1,id:3,add:3,mul:1",
            # default="add:3,mul:1",
            help="Which operator to remove",
        )
        parser.add_argument(
            "--operators_to_not_repeat",
            type=str,
            default="",
            help="Which operator to not repeat",
        )
        parser.add_argument(
            "--max_unary_depth",
            type=int,
            default=7,
            help="Max number of operators inside unary",
        )
        parser.add_argument(
            "--required_operators",
            type=str,
            default="",
            help="Which operator to remove",
        )
        parser.add_argument(
            "--extra_unary_operators",
            type=str,
            default="",
            help="Extra unary operator to add to data generation",
        )
        parser.add_argument(
            "--extra_binary_operators",
            type=str,
            default="",
            help="Extra binary operator to add to data generation",
        )
        parser.add_argument(
            "--extra_constants",
            type=str,
            default=None,
            help="Additional int constants floats instead of ints",
        )

        parser.add_argument("--min_dimension", type=int, default=1)
        parser.add_argument("--max_dimension", type=int, default=2)
        parser.add_argument("--max_masked_variables", type=int, default=0)
        parser.add_argument(
            "--enforce_dim",
            type=bool_flag,
            default=False,
            help="should we enforce that we get as many examples of each dim ?",
        )
        parser.add_argument(
            "--use_controller",
            type=bool_flag,
            default=True,
            help="should we enforce that we get as many examples of each dim ?",
        )
        parser.add_argument(
            "--train_noise_gamma",
            type=float,
            default=0.0,
            help="Should we train with additional output noise",
        )
        parser.add_argument(
            "--eval_noise_gamma",
            type=float,
            default=0.0,
            help="Should we evaluate with additional output noise",
        )
        parser.add_argument(
            "--float_precision",
            type=int,
            default=3,
            help="Number of digits in the mantissa",
        )
        parser.add_argument(
            "--float_descriptor_length",
            type=int,
            default=3,
            help="Type of encoding for floats",
        )
        parser.add_argument("--max_exponent", type=int, default=100, help="Maximal order of magnitude")
        parser.add_argument("--max_trajectory_value", type=float, default=1e2, help="Maximal value in trajectory")
        parser.add_argument(
            "--discard_stationary_trajectory_prob", type=float, default=0.9, help="Probability to discard stationary trajectories"
        )
        parser.add_argument(
            "--max_prefactor",
            type=int,
            default=20,
            help="Maximal order of magnitude in prefactors",
        )
        parser.add_argument(
            "--max_token_len",
            type=int,
            default=0,
            help="max size of tokenized sentences, 0 is no filtering",
        )
        parser.add_argument(
            "--tokens_per_batch",
            type=int,
            default=10000,
            help="max number of tokens per batch",
        )
        parser.add_argument(
            "--pad_to_max_dim",
            type=bool_flag,
            default=True,
            help="should we pad inputs to the maximum dimension?",
        )

        parser.add_argument(
            "--use_two_hot",
            type=bool_flag,
            default=False,
            help="Whether to use two hot embeddings",
        )
        parser.add_argument(
            "--max_int",
            type=int,
            default=10,
            help="Maximal integer in symbolic expressions",
        )
        parser.add_argument(
            "--min_binary_ops_per_dim",
            type=int,
            default=1,
            help="Min number of binary operators per input dimension",
        )
        parser.add_argument(
            "--max_binary_ops_per_dim",
            type=int,
            default=5,
            help="Max number of binary operators per input dimension",
        )
        parser.add_argument("--min_unary_ops_per_dim", type=int, default=0, help="Min number of unary operators")
        parser.add_argument(
            "--max_unary_ops_per_dim",
            type=int,
            default=3,
            help="Max number of unary operators",
        )
        parser.add_argument(
            "--min_op_prob",
            type=float,
            default=0.01,
            help="Minimum probability of generating an example with given n_op, for our curriculum strategy",
        )
        parser.add_argument("--max_points", type=int, default=200, help="Max number of terms in the series")
        parser.add_argument("--min_points", type=int, default=50, help="Min number of terms per dim")

        parser.add_argument(
            "--prob_const",
            type=float,
            default=0.0,
            help="Probability to generate const in leafs",
        )
        parser.add_argument(
            "--prob_prefactor",
            type=float,
            default=1,
            help="Probability to generate prefactor",
        )
        parser.add_argument(
            "--reduce_num_constants",
            type=bool_flag,
            default=True,
            help="Use minimal amount of constants in eqs",
        )
        parser.add_argument(
            "--use_skeleton",
            type=bool_flag,
            default=False,
            help="should we use a skeleton rather than functions with constants",
        )

        parser.add_argument(
            "--prob_rand",
            type=float,
            default=0.0,
            help="Probability to generate n in leafs",
        )

        # ODE
        parser.add_argument(
            "--time_range",
            type=float,
            default=10.0,
            help="Time range for ODE integration",
        )
        parser.add_argument(
            "--prob_t",
            type=float,
            default=0.0,
            help="Probability to generate n in leafs",
        )
        parser.add_argument(
            "--train_subsample_ratio",
            type=float,
            default=0.5,
            help="Ratio of timesteps to remove",
        )
        parser.add_argument(
            "--eval_subsample_ratio",
            type=float,
            default=0,
            help="Ratio of timesteps to remove",
        )
        parser.add_argument(
            "--ode_integrator",
            type=str,
            default="solve_ivp",
            help="ODE integrator to use",
        )
        parser.add_argument(
            "--init_scale",
            type=float,
            default=1.0,
            help="Scale for initial conditions",
        )
        parser.add_argument(
            "--fixed_init_scale",
            type=bool_flag,
            default=False,
            help="Fix the init scale",
        )
