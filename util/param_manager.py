import sys
import argparse
from argparse import Namespace
import yaml
import re

class _ExplicitArgSentinel:
    pass

class ArgTreeManager:
    """
    Makes the following assumptions:
    - Keys in the namespace are valid identifiers i.e. they do not contain periods. The argparse
      library does not check for this.
    - Identifiers are unique, even across hierarchy levels. So an intended namespace that contains
      both args.foo and args.e.foo is invalid.
    - Parameters are all 'simple' types. That is, integers, floats, strings, bools, None type,
      and the composite types lists, dicts, and sets. No object types are permitted. Tuples are
      encoded as lists.
      This assumption is part of a precaution to avoid executing arbitrary code from the yaml file.
      We can probably trust these files, but the precaution doesn't cost much since parameters are
      probably simple anyway.
    - Parameters are all keyword arguments, e.g. --foo FOO, not positional arguments.
    """
 
    sentinel = _ExplicitArgSentinel()

    def __init__(self, parent=None, name=None, parser=None):
        if callable(parser):
            self._parser_constructor = parser
            self.argparser = None
        else:
            self._parser_constructor = None
            self.argparser = parser
        self.parent = parent
        self.name = name
        self.children = {}
        self.children_long_names = {}
        self.explicit = {}

        self.args = None
        self.loaded_arg_string = ""

    def set_parser(self, parser):
        if callable(parser):
            self._parser_constructor = parser
            self.argparser = None
        else:
            self._parser_constructor = None
            self.argparser = parser

    def copy(self, new_parent=None):
        manager_copy = ArgTreeManager()
        manager_copy._parser_constructor = self._parser_constructor
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

    def print_help(self):
        self.argparser.print_help()

    def create_child(self, name, long_name=None, parser=None):
        new_child = ArgTreeManager(parent=self, name=name, parser=parser)
        self.children[name] = new_child
        self.children_long_names[name] = long_name
        return new_child

    def get_child(self, name, long_name=None):
        return self.children[name]

    def get_parent(self):
        return self.parent

    def check_explicit(self, arg_name):
        if '.' in arg_name:
            child_name, _, rest_of_name = arg_name.partition('.')
            return self.children[child_name].check_explicit(rest_of_name)
        else:
            return self.explicit[arg_name]

    def set_explicit(self, *params, is_explicit=True):
        """
        Set whether a parameter is considered explicitly specified.
        You might do this if, for example, a different parameter is explicitly specified that
        requires other defaults; those defaults are also explicitly specified, in a way.

        Multiple parameter names may be passed.
        """
        for param in params:
            if '.' in param:
                child_name, _, rest_of_name = param.partition('.')
                self.children[child_name].set_explicit(rest_of_name, is_explicit=is_explicit)
            else:
                self.explicit[param] = is_explicit

    def parse_args(self, *args, **kwargs):
        args, rest = self.parse_known_args(*args, **kwargs)
        if len(rest) > 0:
            raise Exception("Param manager couldn't parse these arguments: "
                    + " ".join(rest))
        else:
            return args

    def parse_known_args(self, arg_string=None, parent_args=None):
        if arg_string is None:
            arg_string = sys.argv[1:]

        self.loaded_arg_string = arg_string

        if self._parser_constructor is not None:
            if parent_args is None:
                parent_args = Namespace()
            self.argparser = self._parser_constructor(parent_args)

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
        for child_name, child in self.children.items():
            # This allows for conditional sub-parsing. The child parser may now have a different
            # set of arguments depending on the arguments we've already parsed.
            child_args, arg_string = child.parse_known_args(arg_string, parent_args=self.args)
            args_dict[child_name] = child_args

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
        yaml_encoded = yaml.safe_dump(main_args_dict, sort_keys=False)

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

    def load_from_dict(self, load_dict, parent_args=None):
        """
        Load new parameters from a nested dict.

        Any parameters that were already passeded explicitly will not be overridden.
        Any parameters missing from the dict will keep their current values. Any parameters in the
        dict that are not in the existing parameters will be ignored.

        Parameters
        ----------
        load_dict : dict
            Nested dictionary of parameters to load.
        parent_args : Namespace
            The args namespace of the parent ArgTreeManager. Used internally, but also potentially
            useful for loading a subtree based on changes to parent parameters.
        """
        # Reparse in case we're using a conditional parser and parent_args have changed.
        if self._parser_constructor is not None:
            if parent_args is None:
                parent_args = self.parent.args
            self.argparser = self._parser_constructor(parent_args)
            self.args, _ = self.argparser.parse_known_args(self.loaded_arg_string)
        else:
            self.args, _ = self.argparser.parse_known_args(self.loaded_arg_string)
        # Relink parent args to our args.
        if self.parent is not None:
            parent_args_dict = vars(self.parent.args)
            parent_args_dict[self.name] = self.args

        args_dict = vars(self.args)
        my_params = set(args_dict.keys())
        my_children = set(self.children.keys())
        load_dict_names = set(load_dict.keys())

        params_only_in_self = my_params - load_dict_names
        child_only_in_self = my_children - load_dict_names
        only_in_loaded = load_dict_names - (my_params | my_children)

        if len(params_only_in_self) > 0:
            print("Param: Some parameters were not found in loaded file: "
                    + ", ".join([f"{name}: {args_dict[name]}" for name in params_only_in_self]))
        if len(child_only_in_self) > 0:
            print("Param: Entire parameter families were missing from the loaded file;"
                    + " using defaults: " + ", ".join(child_only_in_self))
            for child_name in child_only_in_self:
                args_dict[child_name] = self.children[child_name].args
        if len(only_in_loaded) > 0:
            print("Param: Some parameters in loaded file were not recognized and ignored: "
                    + ", ".join(only_in_loaded))

        params_in_both = my_params & load_dict_names
        children_in_both = my_children & load_dict_names
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
                args_dict[child_name] = self.children[child_name].args

    def load_keys(self, load_dict, keys):
        args_dict = vars(self.args)
        for key in keys:
            if key not in self.children:
                args_dict[key] = load_dict[key]
            else:
                args_dict[ley] = self.children[key].load_from_dict(load_dict[key])

    def init_from_dict(self, load_dict, children_names=None):
        """
        Initialize parameters from a dict instead of parsing them.

        All parameters will be marked as not explicit. Unlike load_from_dict(), all parameters in
        the dict will be used. The parser, if set, will be set to None.

        Parameters
        ----------
        load_dict : dict
            Dict of parameters to load.
        children_names : list of str
            Explicit list of children names. If left as None, all nested dicts will be assumed
            children. If populated, only nested dicts with names in children_names will create
            children. Use '.'s for nested children, e.g. children_names=['foo', 'foo.bar'].

        Returns
        -------
        args : nested Namespace
            The loaded nested namespace, self.args.
        """
        self.argparser = None
        self._parser_constructor = None

        if children_names is None:
            grandchildren_names = None
        else:
            nested_names = [name for name in children_names if '.' in name]
            local_names = [name for name in children_names if '.' not in name]
            grandchildren_names = {name:[] for name in local_names}
            for nested_name in nested_names:
                child_name, _, grandchild_name = nested_name.partition('.')
                grandchildren_names[child_name] = grandchild_name
            children_names = local_names

        args_dict = {}
        for name, value in load_dict.items():
            if type(value) is dict and (children_names is None or name in children_names):
                child = self.create_child(name)
                child.init_from_dict(value, children_names=grandchildren_names[name])
                args_dict[name] = child.args
            else:
                args_dict[name] = value
                self.explicit[name] = False
 
        self.args = Namespace(**args_dict)

        return self.args


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
