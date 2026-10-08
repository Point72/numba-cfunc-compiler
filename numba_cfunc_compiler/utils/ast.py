import ast

__all__ = [
    "AST",
    "add_statement_to_list",
]


def add_statement_to_list(node_list: list[ast.stmt], node: ast.stmt | list[ast.stmt] | None) -> None:
    if isinstance(node, list):
        node_list.extend(node)
    elif node is not None:
        node_list.append(node)


class AST:
    @staticmethod
    def array_access(name: str, index) -> ast.Subscript:
        idx_expr = index if isinstance(index, ast.AST) else ast.Constant(value=index)
        return ast.Subscript(
            value=ast.Name(id=name, ctx=ast.Load()),
            slice=idx_expr,
            ctx=ast.Load(),
        )

    @staticmethod
    def function_call(name: str, *args) -> ast.Call:
        return ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=list(args),
            keywords=[],
        )

    @staticmethod
    def assignment(target, value: ast.AST) -> ast.Assign:
        """Create an assignment to a target which can be a str Name or an AST (e.g., Subscript)."""
        if isinstance(target, str):
            lhs = ast.Name(id=target, ctx=ast.Store())
        elif isinstance(target, ast.Name):
            lhs = ast.Name(id=target.id, ctx=ast.Store())
        elif isinstance(target, ast.Subscript):
            lhs = ast.Subscript(value=target.value, slice=target.slice, ctx=ast.Store())
        elif isinstance(target, ast.Attribute):
            lhs = ast.Attribute(value=target.value, attr=target.attr, ctx=ast.Store())
        else:
            lhs = target

        return ast.Assign(targets=[lhs], value=value)

    @staticmethod
    def deref_pointer(name: str) -> ast.Subscript:
        return AST.array_access(name, 0)

    @staticmethod
    def cast_from_voidptr(var_name_ptr: str, numba_type_name: str) -> ast.Call:
        return AST.function_call("cast_voidptr_to_ptr", var_name_ptr, ast.Constant(value=numba_type_name))
