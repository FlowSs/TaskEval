import ast

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.counts = {}
        self.nested_counts = {}

    def visit(self, node):
        node_type = type(node).__name__
        # List of nodes that should not be there
        if isinstance(node, ast.ClassDef):
            raise Exception("ClassDef node detected")

        # List of nodes to skip
        if not(isinstance(node, ast.Module)):
            if self.is_nested(node):
                parent = getattr(node, 'parent', None)
                if parent is None:
                    raise Exception("no parent :(")
                if type(parent).__name__ + '_' + node_type in self.nested_counts:
                    self.nested_counts[type(parent).__name__ + '_' +node_type] += 1
                else:
                    self.nested_counts[type(parent).__name__ + '_' +node_type] = 1
            else:
                if node_type in self.counts:
                    self.counts[node_type] += 1
                else:
                   self.counts[node_type] = 1            
            
        self.generic_visit(node)


    def is_nested(self, node):
        parent = getattr(node, 'parent', None)
        while parent:
            # Skipping ast.Module as a parent
            if isinstance(parent, ast.Module):
                parent = getattr(parent, 'parent', None)
                continue

            if not(isinstance(node, ast.FunctionDef)):
               # If we have a non FunctionDef node, it is nested only if it's either within non FunctionDef structure or, if it's a FunctionDef structure, said structure is itself nested
               if not(isinstance(parent, ast.FunctionDef)) or (isinstance(parent, ast.FunctionDef) and not(isinstance(getattr(parent, 'parent', None), ast.Module)) ):
                return True
            else:
               if isinstance(parent, ast.FunctionDef):
                return True
            parent = getattr(parent, 'parent', None)
        return False

def count_code_elements(code):
    tree = ast.parse(code)

    # Attach parent nodes to each node in the AST for easy nested detection
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    analyzer = CodeAnalyzer()
    analyzer.visit(tree)

    return {
        'counts': analyzer.counts,
        'nested_counts': analyzer.nested_counts
    }

