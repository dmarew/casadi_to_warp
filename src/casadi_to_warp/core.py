import casadi as ca
import re
import os
import sys
import importlib.util


class CasadiToWarp:
    """
    Transpiles CasADi SX functions to NVIDIA Warp kernels using Mixed Precision.

    Precision Scheme:
    - Inputs:  float32 (Memory efficient)
    - Compute: float64 (High precision, stability)
    - Outputs: float32 (Memory efficient)
    """

    def __init__(
        self,
        casadi_func: ca.Function,
        function_name="casadi_kernel_gen",
        output_dir=".",
        safe_math=True,
    ):
        self.func = casadi_func
        self.name = function_name
        self.dtype_str = "wp.float64"  # Internal compute
        self.io_dtype_str = "wp.float32"  # I/O
        self.source_code = ""
        self.output_dir = output_dir

        if safe_math:
            self.math_map = {
                r"\bcasadi_sqrt\(": "safe_sqrt(",
                r"\bsqrt\(": "safe_sqrt(",
                r"\basin\(": "safe_asin(",
                r"\bacos\(": "safe_acos(",
                r"\bcasadi_sq\(": "sq(",
                r"\bsq\(": "sq(",
                r"\bcasadi_pow\(": "wp.pow(",
                r"\bpow\(": "wp.pow(",
                r"\bexp\(": "wp.exp(",
                r"\blog\(": "wp.log(",
                r"\blog10\(": "wp.log10(",
                r"\bsin\(": "wp.sin(",
                r"\bcos\(": "wp.cos(",
                r"\btan\(": "wp.tan(",
                r"\batan\(": "wp.atan(",
                r"\batan2\(": "wp.atan2(",
                r"\bsinh\(": "wp.sinh(",
                r"\bcosh\(": "wp.cosh(",
                r"\btanh\(": "wp.tanh(",
                r"\basinh\(": "wp.asinh(",
                r"\bacosh\(": "wp.acosh(",
                r"\batanh\(": "wp.atanh(",
                r"\bcasadi_fabs\(": "wp.abs(",
                r"\bfabs\(": "wp.abs(",
                r"\babs\(": "wp.abs(",
                r"\bfloor\(": "wp.floor(",
                r"\bceil\(": "wp.ceil(",
                r"\bcasadi_sign\(": "wp.sign(",
                r"\bsign\(": "wp.sign(",
                r"\bcopysign\(": "c_copysign(",
                r"\bcasadi_fmod\(": "c_fmod(",
                r"\bfmod\(": "c_fmod(",
                r"\bcasadi_fmin\(": "wp.min(",
                r"\bfmin\(": "wp.min(",
                r"\bcasadi_fmax\(": "wp.max(",
                r"\bfmax\(": "wp.max(",
                r"\bcasadi_erf\(": "wp.erf(",
                r"\berf\(": "wp.erf(",
            }
        else:
            self.math_map = {
                r"\bcasadi_sqrt\(": "wp.sqrt(",
                r"\bsqrt\(": "wp.sqrt(",
                r"\basin\(": "wp.asin(",
                r"\bacos\(": "wp.acos(",
                r"\bcasadi_sq\(": "sq(",
                r"\bsq\(": "sq(",
                r"\bcasadi_pow\(": "wp.pow(",
                r"\bpow\(": "wp.pow(",
                r"\bexp\(": "wp.exp(",
                r"\blog\(": "wp.log(",
                r"\blog10\(": "wp.log10(",
                r"\bsin\(": "wp.sin(",
                r"\bcos\(": "wp.cos(",
                r"\btan\(": "wp.tan(",
                r"\batan\(": "wp.atan(",
                r"\batan2\(": "wp.atan2(",
                r"\bsinh\(": "wp.sinh(",
                r"\bcosh\(": "wp.cosh(",
                r"\btanh\(": "wp.tanh(",
                r"\basinh\(": "wp.asinh(",
                r"\bacosh\(": "wp.acosh(",
                r"\batanh\(": "wp.atanh(",
                r"\bcasadi_fabs\(": "wp.abs(",
                r"\bfabs\(": "wp.abs(",
                r"\babs\(": "wp.abs(",
                r"\bfloor\(": "wp.floor(",
                r"\bceil\(": "wp.ceil(",
                r"\bcasadi_sign\(": "wp.sign(",
                r"\bsign\(": "wp.sign(",
                r"\bcopysign\(": "c_copysign(",
                r"\bcasadi_fmod\(": "c_fmod(",
                r"\bfmod\(": "c_fmod(",
                r"\bcasadi_fmin\(": "wp.min(",
                r"\bfmin\(": "wp.min(",
                r"\bcasadi_fmax\(": "wp.max(",
                r"\bfmax\(": "wp.max(",
                r"\bcasadi_erf\(": "wp.erf(",
                r"\berf\(": "wp.erf(",
            }

    def _generate_c_source(self):
        filename = f"{self.name}_tmp.c"
        opts = {"main": False, "mex": False, "with_header": False}
        self.func.generate(filename, opts)
        with open(filename, "r") as f:
            c_code = f.read()
        if os.path.exists(filename):
            os.remove(filename)
        return c_code

    def _parse_body(self, c_code):
        func_pattern = re.compile(
            r"(?:static\s+)?(?:int|void)\s+\w+\s*\(\s*const\s+casadi_real\*\*?\s*arg,\s*casadi_real\*\*?\s*res.*?\)\s*\{(.*?)\}",
            re.DOTALL,
        )
        match = func_pattern.search(c_code)
        if not match:
            func_pattern_simple = re.compile(
                r"void \w+\(const casadi_real\*\* arg, casadi_real\*\* res\)\s*\{(.*?)\}",
                re.DOTALL,
            )
            match = func_pattern_simple.search(c_code)
            if not match:
                raise ValueError("Could not parse CasADi C output.")

        body = match.group(1)
        lines = body.split("\n")
        parsed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("//", "#", "return", "if (sz_", "if (arg")):
                continue
            if line.startswith("casadi_real ") and "=" not in line:
                continue

            # 0. Strip Inline Comments
            if "//" in line:
                line = line.split("//")[0].strip()

            # 1. Cleanup specific CasADi artifacts
            line = re.sub(
                r"arg\[(\d+)\]\s*\?\s*arg\[\1\]\[(\d+)\]\s*:\s*0", r"arg[\1][\2]", line
            )

            # 2. Logic Operators
            line = line.replace(" && ", " and ")
            line = line.replace(" || ", " or ")
            line = re.sub(r"!(?!=)", " not ", line)

            # 3. Strip Outer Parentheses for Assignment (Precise)
            if "=" in line and line.rstrip().endswith(");"):
                match = re.search(r"=\s*(\(.*\));", line)
                if match:
                    content = match.group(1)
                    if content.count("(") == content.count(")"):
                        start, end = match.span(1)
                        line = line[:start] + content[1:-1] + line[end:]

            # 4. Ternary Conversion
            if "?" in line and ":" in line:

                def ternary_replacer(m):
                    cond = m.group(1).strip()
                    true_val = m.group(2).strip()
                    false_val = m.group(3).strip()

                    if cond.startswith("(") and false_val.endswith(")"):
                        if cond.count("(") > cond.count(")") and false_val.count(
                            ")"
                        ) > false_val.count("("):
                            cond = cond[1:].strip()
                            false_val = false_val[:-1].strip()

                    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", cond):
                        cond = f"({cond} != 0.0)"
                    elif cond.startswith("(") and cond.endswith(")"):
                        inner = cond[1:-1]
                        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", inner):
                            cond = f"({cond} != 0.0)"
                    return f"= {true_val} if ({cond}) else {false_val};"

                line = re.sub(
                    r"=\s*(.*?)\s*\?\s*(.*?)\s*:\s*(.*?);", ternary_replacer, line
                )

            # 5. Boolean Casting
            if "=" in line and " if " not in line and " else " not in line:
                if any(
                    op in line
                    for op in ["<", ">", "==", "!=", " and ", " or ", " not "]
                ):
                    parts = line.split("=", 1)
                    lhs = parts[0]
                    rhs = parts[1].strip()
                    semicolon = ";" if rhs.endswith(";") else ""
                    if semicolon:
                        rhs = rhs[:-1]
                    line = f"{lhs}= FloatT({rhs}){semicolon}"

            # 6. Output Checks
            if "if (res" in line:
                assign_match = re.search(r"res\[\d+\]\[\d+\]\s*=[^;]+;", line)
                if assign_match:
                    line = assign_match.group(0)

            # 7. Variable Names
            line = re.sub(r"arg\[(\d+)\]", r"inputs_\1", line)
            line = re.sub(r"res\[(\d+)\]", r"outputs_\1", line)

            # 8. Apply Math Map
            for pattern, replacement in self.math_map.items():
                line = re.sub(pattern, replacement, line)

            # 9. Fix Array Indexing & Input Cast
            def index_replacer(m):
                name = f"{m.group(1)}{m.group(2)}"
                idx = m.group(3)
                ref = f"{name}[tid, __IDX_{idx}__]"
                if "inputs" in name:
                    return f"FloatT({ref})"
                return ref

            line = re.sub(r"(inputs_|outputs_)(\d+)\[(\d+)\]", index_replacer, line)

            # 10. OUTPUT CAST: float64 -> float32
            if line.strip().startswith("outputs_"):
                parts = line.split("=", 1)
                lhs = parts[0]
                rhs = parts[1].strip()
                has_semicolon = rhs.endswith(";")
                if has_semicolon:
                    rhs = rhs[:-1]

                # Cleanup lingering closing paren if any
                if rhs.endswith(")") and "(" not in rhs:
                    rhs = rhs[:-1]

                line = f"{lhs}= wp.float32({rhs})" + (";" if has_semicolon else "")

            # 11. Cast Numeric Literals
            numeric_pattern = r"(?<![\w])(\d+(\.\d*)?([eE][+-]?\d+)?)"
            line = re.sub(numeric_pattern, r"FloatT(\1)", line)

            # Restore Indices
            line = re.sub(r"__IDX_(\d+)__", r"\1", line)

            if line.endswith(";"):
                line = line[:-1]
            parsed_lines.append(line)
        return parsed_lines

    def transpile(self):
        c_code = self._generate_c_source()
        body_lines = self._parse_body(c_code)

        indent = "    "
        python_src = []
        python_src.append("import warp as wp")
        python_src.append("")
        python_src.append(f"FloatT = {self.dtype_str}")
        python_src.append("")

        # Helpers
        python_src.append("@wp.func")
        python_src.append("def sq(x: FloatT):")
        python_src.append(f"{indent}return x * x")
        python_src.append("")

        # --- SAFE HELPERS with Explicit Casting ---

        python_src.append("@wp.func")
        python_src.append("def safe_sqrt(x: FloatT):")
        # Cast 1.0e-12 to FloatT to match x
        python_src.append(f"{indent}return wp.sqrt(wp.max(x, FloatT(1.0e-12)))")
        python_src.append("")

        python_src.append("@wp.func")
        python_src.append("def safe_acos(x: FloatT):")
        python_src.append(
            f"{indent}return wp.acos(wp.clamp(x, FloatT(-0.9999999), FloatT(0.9999999)))"
        )
        python_src.append("")

        python_src.append("@wp.func")
        python_src.append("def safe_asin(x: FloatT):")
        python_src.append(
            f"{indent}return wp.asin(wp.clamp(x, FloatT(-0.9999999), FloatT(0.9999999)))"
        )
        python_src.append("")

        python_src.append("@wp.func")
        python_src.append("def c_copysign(x: FloatT, y: FloatT):")
        python_src.append(
            f"{indent}return wp.abs(x) if (y >= FloatT(0.0)) else -wp.abs(x)"
        )
        python_src.append("")

        python_src.append("@wp.func")
        python_src.append("def c_fmod(x: FloatT, y: FloatT):")
        python_src.append(f"{indent}return x - wp.trunc(x / y) * y")
        python_src.append("")

        n_in = self.func.n_in()
        n_out = self.func.n_out()

        args = []
        for i in range(n_in):
            args.append(f"inputs_{i}: wp.array(dtype={self.io_dtype_str}, ndim=2)")
        for i in range(n_out):
            args.append(f"outputs_{i}: wp.array(dtype={self.io_dtype_str}, ndim=2)")

        arg_str = ", ".join(args)
        python_src.append("@wp.kernel")
        python_src.append(f"def {self.name}({arg_str}):")
        python_src.append(f"{indent}tid = wp.tid()")

        for line in body_lines:
            python_src.append(f"{indent}{line}")

        self.source_code = "\n".join(python_src)
        return self.source_code

    def load_kernel(self):
        if not self.source_code:
            self.transpile()

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        module_filename = os.path.join(self.output_dir, f"{self.name}.py")
        with open(module_filename, "w") as f:
            f.write(self.source_code)

        spec = importlib.util.spec_from_file_location(self.name, module_filename)
        if spec is None:
            raise ImportError(f"Could not load generated kernel from {module_filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.name] = module
        spec.loader.exec_module(module)
        return getattr(module, self.name)
