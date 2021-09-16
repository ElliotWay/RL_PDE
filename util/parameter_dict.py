import sys
import argparse
from argparse import Namespace

class HierarchyParser:
    def __init__(self, parent=None):
        self.argparser = None
        self.parent = parent
        self.children = {}

    def set_parser(self, argparser):
        self.argparser = argparser

    def get_child(self, name):
        if name in self.children:
            return self.children[name]
        else:
            new_child = HierarchyParser(parent=self)
            self.children[name] = new_child
            return new_child

    def get_parent(self):
        return self.parent

    def parse(self, arg_string=None):
        if arg_string is None:
            arg_string = sys.argv[1:]
        if self.argparser is not None:
            main_args, arg_string = self.argparser.parse_known_args(arg_string)
        else:
            main_args = Namespace()
        # Note: this is not a deep copy! Writing to main_args_dict will also affect main_args.
        main_args_dict = vars(main_args)
        for child, child_parser in self.children.items():
            child_args, arg_string = child_parser.parse(arg_string)
            main_args_dict[child] = child_args

        return main_args, arg_string

    @staticmethod
    def serialize_nested_args(args):
        # How do we keep track of explicit arguments?

    

if __name__ == "__main__":
    child1_args = argparse.ArgumentParser()
    child1_args.add_argument('--child1_arg')

    child2_args = argparse.ArgumentParser()
    child2_args.add_argument('--child2_arg')

    child3_args = argparse.ArgumentParser()
    child3_args.add_argument('--child3_arg')

    root_parser = HierarchyParser()
    child1_parser = root_parser.get_child('child1')
    child1_parser.set_parser(child1_args)

    child2_parser = root_parser.get_child('child2')
    child2_parser.set_parser(child2_args)

    child3_parser = child1_parser.get_child('child3')
    child3_parser.set_parser(child3_args)

    args, rest = root_parser.parse()
    print("all args:")
    print(args)

    print("args.child1.child1_arg", args.child1.child1_arg)
    print("args.child2.child2_arg", args.child2.child2_arg)
    print("args.child1.child3.child3_arg", args.child1.child3.child3_arg)
