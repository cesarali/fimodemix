import numpy as np
import scipy
import scipy.special


math_constants = ["e", "pi", "euler_gamma", "CONSTANT"]


class Node:
    def __init__(self, value, params, children=None):
        self.value = value
        self.children = children if children else []
        self.params = params

    def push_child(self, child):
        self.children.append(child)

    def prefix(self, skeleton=False):
        s = str(self.value)
        if skeleton:
            try:
                float(s)
                s = "CONSTANT"
            except:
                pass
        for c in self.children:
            s += "," + c.prefix(skeleton=skeleton)
        return s

    # export to latex qtree format: prefix with \Tree, use package qtree
    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self, skeleton=False):
        s = str(self.value)
        if skeleton:
            try:
                float(s)
                s = "CONSTANT"
            except:
                pass
        nb_children = len(self.children)
        if nb_children == 0:
            return s
        if nb_children == 1:
            if s == "pow2":
                s = "(" + self.children[0].infix(skeleton=skeleton) + ")**2"
            elif s == "inv":
                s = "1/(" + self.children[0].infix(skeleton=skeleton) + ")"
            elif s == "pow3":
                s = "(" + self.children[0].infix(skeleton=skeleton) + ")**3"
            elif s == "sqrt":
                if self.params.use_abs:
                    s = "sqrt(abs(" + self.children[0].infix(skeleton=skeleton) + "))"
                else:
                    s = "sqrt(" + self.children[0].infix(skeleton=skeleton) + ")"
            elif s == "log":
                if self.params.use_abs:
                    s = "log(abs(" + self.children[0].infix(skeleton=skeleton) + "))"
                else:
                    s = "log(" + self.children[0].infix(skeleton=skeleton) + ")"
            elif s == "dexp":
                s = "exp(-abs(" + self.children[0].infix(skeleton=skeleton) + "))"
            else:
                s = s + "(" + self.children[0].infix(skeleton=skeleton) + ")"
            return s
        else:
            if s == "add":
                return self.children[0].infix(skeleton=skeleton) + " + " + self.children[1].infix(skeleton=skeleton)
            if s == "sub":
                return self.children[0].infix(skeleton=skeleton) + " - " + self.children[1].infix(skeleton=skeleton)
            if s == "pow":
                res = "(" + self.children[0].infix(skeleton=skeleton) + ")**"
                res += "" + self.children[1].infix(skeleton=skeleton)
                return res
            elif s == "mul":
                res = (
                    "(" + self.children[0].infix(skeleton=skeleton) + ")"
                    if self.children[0].value in ["add", "sub"]
                    else (self.children[0].infix(skeleton=skeleton))
                )
                res += " * "
                res += (
                    "(" + self.children[1].infix(skeleton=skeleton) + ")"
                    if self.children[1].value in ["add", "sub"]
                    else (self.children[1].infix(skeleton=skeleton))
                )
                return res
            elif s == "div":
                res = (
                    "(" + self.children[0].infix(skeleton=skeleton) + ")"
                    if self.children[0].value in ["add", "sub"]
                    else (self.children[0].infix(skeleton=skeleton))
                )
                res += " / "
                res += (
                    "(" + self.children[1].infix(skeleton=skeleton) + ")"
                    if self.children[1].value in ["add", "sub"]
                    else (self.children[1].infix(skeleton=skeleton))
                )
                return res

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        # infix a default print
        return self.prefix()

    def __repr__(self):
        # infix a default print
        return str(self)

    def val(self, x, t, deterministic=True):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(self.children) == 0:
            if str(self.value).startswith("x_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return x[:, dim]
            elif str(self.value) == "t":
                return t
            elif str(self.value) == "rand":
                if deterministic:
                    return np.zeros((x.shape[0],))
                return np.random.randn(x.shape[0])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value)) * np.ones((x.shape[0],))
            else:
                return float(self.value) * np.ones((x.shape[0],))

        if self.value == "add":
            return self.children[0].val(x, t) + self.children[1].val(x, t)
        if self.value == "sub":
            return self.children[0].val(x, t) - self.children[1].val(x, t)
        if self.value == "mul":
            m1, m2 = self.children[0].val(x, t), self.children[1].val(x, t)
            try:
                return m1 * m2
            except Exception:
                # print(e)
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow":
            m1, m2 = self.children[0].val(x, t), self.children[1].val(x, t)
            try:
                return np.power(m1, m2)
            except Exception:
                # print(e)
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "max":
            return np.maximum(self.children[0].val(x, t), self.children[1].val(x, t))
        if self.value == "min":
            return np.minimum(self.children[0].val(x, t), self.children[1].val(x, t))

        if self.value == "div":
            denominator = self.children[1].val(x, t)
            denominator[denominator == 0.0] = np.nan
            try:
                return self.children[0].val(x, t) / denominator
            except Exception:
                # print(e)
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "inv":
            denominator = self.children[0].val(x, t)
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator
            except Exception:
                # print(e)
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "log":
            numerator = self.children[0].val(x, t)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log(numerator)
            except Exception:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans

        if self.value == "dexp":
            numerator = self.children[0].val(x, t)
            numerator[numerator <= 0.0] *= -1

            try:
                return np.log(numerator)
            except Exception:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans

        if self.value == "sqrt":
            numerator = self.children[0].val(x, t)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator < 0.0] = np.nan
            try:
                return np.sqrt(numerator)
            except Exception:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow2":
            numerator = self.children[0].val(x, t)
            try:
                return numerator**2
            except Exception:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow3":
            numerator = self.children[0].val(x, t)
            try:
                return numerator**3
            except Exception:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "abs":
            return np.abs(self.children[0].val(x, t))
        if self.value == "sign":
            return (self.children[0].val(x, t) >= 0) * 2.0 - 1.0
        if self.value == "step":
            x = self.children[0].val(x, t)
            return x if x > 0 else 0
        if self.value == "id":
            return self.children[0].val(x, t)
        if self.value == "fresnel":
            return scipy.special.fresnel(self.children[0].val(x, t))[0]
        if self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(x, t))[0]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(self.children[0].val(x, t))
                except Exception:
                    nans = np.empty((x.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(self.children[0].val(x, t))
            assert False, "Could not find function"

    def replace_node_value(self, old_value, new_value):
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)


class NodeList:
    def __init__(self, nodes):
        self.nodes = []
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params

    def infix(self, skeleton=False):
        return " | ".join([node.infix(skeleton=skeleton) for node in self.nodes])

    def __len__(self):
        return sum([len(node) for node in self.nodes])

    def prefix(self, skeleton=False):
        return ",|,".join([node.prefix(skeleton=skeleton) for node in self.nodes])

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, xs, t, deterministic=True):
        batch_vals = [np.expand_dims(node.val(np.copy(xs), t, deterministic=deterministic), -1) for node in self.nodes]
        return np.concatenate(batch_vals, -1)

    def replace_node_value(self, old_value, new_value):
        for node in self.nodes:
            node.replace_node_value(old_value, new_value)

    def get_dimension(self):
        return len(self.nodes)
