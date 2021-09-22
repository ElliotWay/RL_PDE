import sys
import argparse
from argparse import Namespace
import yaml
import re

class _ExplicitArgSentinel:
    pass

class ArgTreeManager:
    """
    Assumes that keys in the namespace are valid identifiers and do not contain periods.
    (argparse does not prevent this).
    Also assumes that identifiers are unique, even across hierarchy levels. So args.foo and
    args.e.foo is not valid.
    """
 
    sentinel = _ExplicitArgSentinel()

    def __init__(self, parent=None):
        self.argparser = None
        self.parent = parent
        self.children = {}
        self.children_long_names = {}
        self.explicit = {}

        self.args = None

    def copy(self, new_parent=None):
        manager_copy = ArgTreeManager()
        manager_copy.argparser = self.argparser
        manager_copy.parent = new_parent
        for child_name, child in self.children.items():
            manager_copy.children[child_name] = child.copy(new_parent=manager_copy)
        manager_copy.children_long_names = dict(self.children_long_names)
        manager_copy.explicit = dict(self.explicit)

        manager_copy.args = Namespace(**vars(self.args))
        manager_dict = vars(manager_copy.args)
        for name, value in manager_dict.items():
            if isinstance(value, Namespace):
                manager_dict[name] = manager_copy.children[name]

        return manager_copy

    def set_parser(self, argparser):
        self.argparser = argparser

    def print_help(self):
        self.arg_parser.print_help()

    def get_child(self, name, long_name=None):
        if name in self.children:
            return self.children[name]
        else:
            new_child = ArgTreeManager(parent=self)
            self.children[name] = new_child
            self.children_long_names[name] = long_name
            return new_child

    def get_parent(self):
        return self.parent

    def check_explicit(self, arg_name):
        if '.' in arg_name:
            child_name, _, rest_of_name = arg_name.partition('.')
            return self.children[child_name].check_explicit(rest_of_name)
        else:
            return self.explicit[arg_name]

    def parse_known_args(self, arg_string=None):
        if arg_string is None:
            arg_string = sys.argv[1:]

        if self.argparser is not None:
            main_args, remaining_arg_string = self.argparser.parse_known_args(arg_string)

            # Some hackery to figure out which args were explicitly passed on the command line
            # (or in arg_string), and which args are defaults.
            sentinel_dict = dict(vars(main_args)) # Deep copy.
            for key in sentinel_dict:
                sentinel_dict[key] = ArgTreeManager.sentinel
            sentinel_ns = Namespace(**sentinel_dict)
            # ArgumentParser.parse_known_args() can take a 'namespace' argument. Not only will this
            # populate the passed namespace instead of creating a new one, but anything already in
            # that namespace will override any defaults.
            self.argparser.parse_known_args(arg_string, namespace=sentinel_ns)
            # Now anything in sentinel_ns that is still a sentinel was NOT passed explicitly.
            self.explicit = {key: value is not ArgTreeManager.sentinel for key, value in
                    vars(sentinel_ns).items()}

            arg_string = remaining_arg_string
        else:
            main_args = Namespace()

        self.args = main_args

        # Not a deep copy - changes to args_dict will change self.args.
        args_dict = vars(self.args)
        for child, child_parser in self.children.items():
            child_args, arg_string = child_parser.parse_known_args(arg_string)
            args_dict[child] = child_args

        return self.args, arg_string

    @staticmethod
    def nested_ns_to_dict(args):
        args_dict = dict(vars(args))
        for k, v in args_dict.items():
            if isinstance(v, Namespace):
                args_dict[k] = nested_ns_to_dict(v)
        return args_dict

    def _preorder(self, func, input_arg):
        acc = func(self, input_arg)
        for child in self.children.values():
            acc = child._preorder(func, acc)
        return acc

    def serialize(self, indent=0):
        """
        Convert all of the parameters and nested parameters into a YAML string.
        Comments are added for the long names of each children, and for each explicit parameter.

        Parameters
        ----------
        indent : int
            Level of indentation. Used internally to manager the indentation of children.

        Returns
        -------
        The string representation of the parameters.
        """
        # Encode args at this node into YAML.
        main_args_dict = {k:v for k,v in vars(self.args).items() if k not in self.children}
        yaml_encoded = yaml.dump(main_args_dict)

        # Add comments to the encoding, and possibly additional indentation.
        lines = yaml_encoded.split('\n')
        explicit_variables = [name for name in self.explicit if self.explicit[name]]

        indent_prefix = "  " * indent
        for index, line in enumerate(lines):
            comment = ""
            match = re.match(r"\s*(\w+):", line)
            if match is not None and match.group(1) in explicit_variables:
                comment = " # (Explicit)"
            lines[index] = f"{indent_prefix}{line}{comment}"

        # Add children.
        for child_name, child in self.children.items():
            comment = ("" if self.children_long_names[child_name] is None else
                            f" # {self.children_long_names[child_name]}")
            lines.append(f"{indent_prefix}{child_name}:{comment}")
            lines.append(child.serialize(indent+1))
        return "\n".join(lines)

    def load_from_dict(self, load_dict):
        args_dict = vars(self.args)
        my_names = set(args_dict.keys())
        load_dict_names = set(load_dict.keys())

        only_in_self = my_names - load_dict_names
        params_only_in_self = list(filter(lambda n: n not in self.children, only_in_self))
        child_only_in_self = list(filter(lambda n: n in self.children, only_in_self))
        if len(params_only_in_self) > 0:
            print("Param: Some parameters were not found in loaded file: "
                    + ", ".join([f"{name}: {args_dict[name]}" for name in params_only_in_self]))
        if len(child_only_in_self) > 0:
            print("Param: Entire parameter families were missing from the loaded file;"
                    + " using defaults: " + ", ".join(child_only_in_self))

        only_in_loaded = load_dict_names - my_names
        if len(only_in_loaded) > 0:
            print("Param: Some parameters in loaded file were not recognized and ignored: "
                    + ", ".join(only_in_loaded))

        names_in_both = my_names & load_dict_names
        params_in_both = list(filter(lambda n: n not in self.children, names_in_both))
        children_in_both = list(filter(lambda n: n in self.children, names_in_both))

        explicit_overrides = list(filter(lambda n: self.explicit[n], params_in_both))
        loaded_params = list(filter(lambda n: not self.explicit[n], params_in_both))

        if len(explicit_overrides) > 0:
            print("Param: Explicit parameters overriding loaded parameters: "
                    + ", ".join(explicit_overrides))
        if len(loaded_params) > 0:
            for param in loaded_params:
                args_dict[param] = load_dict[param]

        if len(children_in_both) > 0:
            for child_name in children_in_both:
                self.children[child_name].load_from_dict(load_dict[child_name])


if __name__ == "__main__":
    child1_args = argparse.ArgumentParser()
    child1_args.add_argument('--child1_arg', default=1)

    child2_args = argparse.ArgumentParser()
    child2_args.add_argument('--child2_arg', default=2)

    child3_args = argparse.ArgumentParser()
    child3_args.add_argument('--child3_arg', default=3)

    root_parser = ArgTreeManager()
    child1_parser = root_parser.get_child('child1', "Child1 Long Name")
    child1_parser.set_parser(child1_args)

    child2_parser = root_parser.get_child('child2')
    child2_parser.set_parser(child2_args)

    child3_parser = child1_parser.get_child('child3')
    child3_parser.set_parser(child3_args)

    args, rest = root_parser.parse_known_args()
    print("didn't recognize ", rest)
    print("all args:")
    print(args)

    print("args.child1.child1_arg", args.child1.child1_arg)
    print("args.child2.child2_arg", args.child2.child2_arg)
    print("args.child1.child3.child3_arg", args.child1.child3.child3_arg)

    print("Serialized version:")
    serialized = root_parser.serialize()
    print(serialized)

    args.child1.child1_arg = 100
    args.child2.child2_arg = 200
    args.child1.child3.child3_arg = 300

    print("Changed to:")
    print(root_parser.serialize())

    print("Yaml dict from original serialized version:")
    yaml_dict = yaml.safe_load(serialized)
    print(yaml_dict)
    print("Loading from original serialized version:")
    root_parser.load_from_dict(yaml_dict)
    print(root_parser.serialize())
