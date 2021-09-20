import sys
import argparse
from argparse import Namespace
import yaml

class _ExplicitArgSentinel:
    pass

class HierarchyParser:
    """
    Assumes that keys in the namespace are valid identifiers and do not contain periods.
    (argparse does not prevent this).
    """
 
    sentinel = _ExplicitArgSentinel()

    def __init__(self, parent=None):
        self.argparser = None
        self.parent = parent
        self.children = {}
        self.children_long_names = {}
        self.explicit = {}

        self.args = None

    def set_parser(self, argparser):
        self.argparser = argparser

    def get_child(self, name, long_name=None):
        if name in self.children:
            return self.children[name]
        else:
            new_child = HierarchyParser(parent=self)
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
                sentinel_dict[key] = HierarchyParser.sentinel
            sentinel_ns = Namespace(**sentinel_dict)
            # ArgumentParser.parse_known_args() can take a 'namespace' argument. Not only will this
            # populate the passed namespace instead of creating a new one, but anything already in
            # that namespace will override any defaults.
            self.argparser.parse_known_args(arg_string, namespace=sentinel_ns)
            # Now anything in sentinel_ns that is still a sentinel was NOT passed explicitly.
            self.explicit = {key: value is not HierarchyParser.sentinel for key, value in
                    vars(sentinel_ns).items()}

            arg_string = remaining_arg_string
        else:
            main_args = Namespace()

        # Not a deep copy: writing to main_args_dict will also affect main_args.
        main_args_dict = vars(main_args)
        for child, child_parser in self.children.items():
            child_args, arg_string = child_parser.parse_known_args(arg_string)
            main_args_dict[child] = child_args

        self.args = main_args

        return main_args, arg_string

    @staticmethod
    def nested_ns_to_dict(args):
        args_dict = dict(vars(args))
        for k, v in args_dict.items():
            if isinstance(v, Namespace):
                args_dict[k] = nested_ns_to_dict(v)
        return args_dict

    def serialize(self, indent=0):
        lines = []
        indent_prefix = "  " * indent
        for k, v in vars(self.args).items():
            if not isinstance(v, Namespace):
                comment = "" if not self.explicit[k] else " #(Explicit)"
                lines.append(f"{indent_prefix}{k}: {v}{comment}")
        for child_name, child in self.children.items():
            comment = ("" if self.children_long_names[child_name] is None else
                            f" #{self.children_long_names[child_name]}")
            lines.append(f"{indent_prefix}{child_name}:{comment}")
            lines.append(child.serialize(indent+1))
        return "\n".join(lines)

    def load_from_dict(self, load_dict, override_explicit=True):
        args_dict = vars(self.args)
        my_names = set(args_dict.keys())
        load_dict_names = set(load_dict.keys())
        for name in my_names - load_dict_names:
            if name not in self.children and not self.explicit[name]:
                print(f"Param: {name} not found, cannot load, using default ({args_dict[name]}).")
            elif name in self.children:
                print(f"Param: {name} parameter family not found, cannot load, using defaults.")
        for name in load_dict_names - my_names:
            if not isinstance(load_dict[name], dict):
                print(f"Param: {name} not recognized, cannot load.")
            else:
                print(f"Param: {name} parameter family not recognized, cannot load.")

        for name in my_names & load_dict_names:
            if not name in self.children:
                if not self.explicit[name]:
                    args_dict[name] = load_dict[name]
                else:
                    print(f"Param: Explicit {name} overriding loaded parameter.")
        for name in my_names & load_dict_names:
            if name in self.children:
                self.children[name].load_from_dict(load_dict[name])


if __name__ == "__main__":
    child1_args = argparse.ArgumentParser()
    child1_args.add_argument('--child1_arg', default=1)

    child2_args = argparse.ArgumentParser()
    child2_args.add_argument('--child2_arg', default=2)

    child3_args = argparse.ArgumentParser()
    child3_args.add_argument('--child3_arg', default=3)

    root_parser = HierarchyParser()
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
