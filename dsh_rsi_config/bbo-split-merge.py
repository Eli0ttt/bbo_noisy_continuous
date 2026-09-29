#!/usr/bin/env python3
"""Deterministic AST ownership contract + merge for split-model BBO research."""
from __future__ import annotations
import argparse, ast, copy, json, sys
from pathlib import Path

REQ = {"__init__", "ask", "tell"}

def fail(msg: str):
    raise SystemExit("SPLIT_MERGE_ERROR: " + msg)

def parse(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except Exception as exc:
        fail(f"{path}: {type(exc).__name__}: {exc}")
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Optimizer"), None)
    if cls is None:
        fail(f"{path}: missing Optimizer")
    methods = {n.name:n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not REQ <= methods.keys():
        fail(f"{path}: Optimizer must define {sorted(REQ)}")
    return src, tree, cls, methods

def dump(n):
    return ast.dump(n, annotate_fields=True, include_attributes=False)

def self_calls(fn):
    out=set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id=="self":
            out.add(n.func.attr)
    return out

def reach(methods, start):
    seen=set(); stack=[start]
    while stack:
        name=stack.pop()
        if name in seen: continue
        seen.add(name)
        fn=methods.get(name)
        if fn:
            stack.extend(x for x in self_calls(fn) if x in methods and x not in seen)
    return seen

def self_attrs(node, ctx=None):
    out=set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id=="self":
            if ctx is None or isinstance(n.ctx, ctx):
                out.add(n.attr)
    return out

def imports(tree):
    return [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]

def validate_imports(tree):
    allowed={"numpy","math","statistics","collections","heapq","bisect","itertools","functools","operator","random"}
    for n in imports(tree):
        names=[x.name.split(".")[0] for x in n.names] if isinstance(n,ast.Import) else [str(n.module or "").split(".")[0]]
        bad=[x for x in names if x and x not in allowed]
        if bad: fail("non-stdlib/non-NumPy import not allowed: "+",".join(bad))

def module_defs(tree):
    """Top-level function/class definitions keyed by name, excluding Optimizer."""
    return {
        n.name: n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.name != "Optimizer"
    }

def normalize_new_module_defs(parent_tree, cand_tree, role):
    """Normalize new private top-level helpers into the role namespace."""
    p = module_defs(parent_tree)
    c = module_defs(cand_tree)
    prefix = "_a_" if role == "A" else "_b_"
    rename = {}
    reserved = set(p) | {"Optimizer"}
    for name in c:
        if name in p or name.startswith(prefix):
            continue
        if name.startswith("_") and not name.startswith("__"):
            target = prefix + name.lstrip("_")
            if target in reserved or target in c:
                fail(f"{role}: cannot normalize {name} -> {target}; collision")
            rename[name] = target
            reserved.add(target)
        else:
            fail(f"{role}: new public module definition {name} is forbidden; use {prefix}*")

    if rename:
        class Renamer(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id in rename: node.id=rename[node.id]
                return node
            def visit_FunctionDef(self,node):
                if node.name in rename: node.name=rename[node.name]
                return self.generic_visit(node)
            def visit_AsyncFunctionDef(self,node):
                if node.name in rename: node.name=rename[node.name]
                return self.generic_visit(node)
            def visit_ClassDef(self,node):
                if node.name in rename: node.name=rename[node.name]
                return self.generic_visit(node)
        Renamer().visit(cand_tree)
        ast.fix_missing_locations(cand_tree)
        c=module_defs(cand_tree)

    peer_prefix = "_b_" if role == "A" else "_a_"
    for name,node in p.items():
        other=c.get(name)
        if other is None:
            fail(f"{role}: existing module-level definition {name} cannot be deleted")
        if name.startswith(prefix):
            continue
        if dump(node)!=dump(other):
            fail(f"{role}: module-level definition {name} is frozen")
    for name in c:
        if name not in p and not name.startswith(prefix):
            fail(f"{role}: module helper {name} is outside namespace {prefix}")
        if name.startswith(peer_prefix) and name in p and dump(c[name])!=dump(p[name]):
            fail(f"{role}: peer-owned module helper {name} is frozen")
    return c, rename

def class_assigns(cls):
    out={}
    for n in cls.body:
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t,ast.Name): out[t.id]=n
    return out

def contract(parent: Path):
    _,tree,cls,methods=parse(parent)
    ask=reach(methods,"ask"); tell=reach(methods,"tell")
    a_pref={n for n in methods if n.startswith("_a_")}
    b_pref={n for n in methods if n.startswith("_b_")}
    shared=sorted(((ask & tell)-{"ask","tell","__init__"})-a_pref-b_pref)
    a_owned=sorted((((tell-ask)|{"__init__","tell"})-b_pref)|a_pref)
    b_owned=sorted((((ask-tell)|{"ask"})-a_pref)|b_pref)
    attrs=sorted(self_attrs(cls))
    return {
      "schema":1,
      "parent":str(parent),
      "ownership":{
        "A":{"existing_methods":a_owned,"new_method_prefix":"_a_","may_edit_imports":True},
        "B":{"existing_methods":b_owned,"new_method_prefix":"_b_","may_edit_class_attr":["batch"]},
        "frozen_shared_methods":shared,
        "frozen_rule":"all symbols not explicitly owned are immutable"
      },
      "parent_self_attributes":attrs,
      "compatibility":[
        "Both candidates start as exact copies of parent.py.",
        "A must not edit B-owned/frozen methods; B must not edit A-owned/frozen methods.",
        "A may add _a_* methods; B may add _b_* methods.",
        "B exclusively controls effective batch via the class attribute batch; A must not assign self.batch in A-owned methods.",
        "A may change imports.",
        "Do not delete or rename parent state attributes used by the peer-owned surface.",
        "A new state intended for B is a next-round handoff: after merge it becomes part of the next parent.",
        "B new proposal state should be initialized inside B-owned methods before it is read.",
        "The coordinator merges AST-owned symbols; it never asks either model to paste peer code."
      ]
    }

def node_map(cls):
    return {n.name:n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

def validate_candidate(parent_path, cand_path, role, c):
    _,pt,pc,pm=parse(parent_path)
    _,ct,cc,cm=parse(cand_path)
    validate_imports(ct)
    c_module_defs, module_renames = normalize_new_module_defs(pt, ct, role)
    # Normalization may rewrite Name references inside Optimizer.
    cc = next(n for n in ct.body if isinstance(n, ast.ClassDef) and n.name == "Optimizer")
    cm = {n.name:n for n in cc.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    own=set(c["ownership"][role]["existing_methods"])
    frozen=set(c["ownership"]["frozen_shared_methods"])
    prefix=c["ownership"][role]["new_method_prefix"]
    p_assign=class_assigns(pc); c_assign=class_assigns(cc)

    for name,pn in pm.items():
        cn=cm.get(name)
        if cn is None: fail(f"{role}: deleted existing method {name}")
        if name not in own and dump(pn)!=dump(cn):
            fail(f"{role}: unauthorized edit to method {name}")
    for name,cn in cm.items():
        if name not in pm and not name.startswith(prefix):
            fail(f"{role}: new method {name} must start with {prefix}")

    if role=="A":
        a_batch_writers=[name for name,node in cm.items()
                         if (name in own or name.startswith("_a_"))
                         and "batch" in self_attrs(node,ast.Store)]
        if a_batch_writers:
            fail("A: self.batch is B-owned; remove instance assignment(s) from: "+",".join(sorted(a_batch_writers)))
        # A owns imports, but class attributes are frozen.
        for name,pn in p_assign.items():
            cn=c_assign.get(name)
            if cn is None or dump(pn)!=dump(cn):
                fail(f"A: class attribute {name} is frozen")
        for name in c_assign:
            if name not in p_assign: fail(f"A: cannot add class attribute {name}")
    else:
        # B cannot change imports or class attrs except batch.
        if [dump(x) for x in imports(pt)] != [dump(x) for x in imports(ct)]:
            fail("B: imports are frozen")
        for name,pn in p_assign.items():
            cn=c_assign.get(name)
            if cn is None: fail(f"B: deleted class attribute {name}")
            if name!="batch" and dump(pn)!=dump(cn):
                fail(f"B: class attribute {name} is frozen")
        for name in c_assign:
            if name not in p_assign and name!="batch":
                fail(f"B: cannot add class attribute {name}")

    # Preserve peer-visible parent state. This catches the common merge break:
    # A renames/deletes state that B's ask path still expects.
    parent_attrs=set(c["parent_self_attributes"])
    cand_attrs=self_attrs(cc)
    if role=="A":
        b_existing=set(c["ownership"]["B"]["existing_methods"])
        a_existing=set(c["ownership"]["A"]["existing_methods"])
        parent_method_names=set(pm)
        peer_reads=set()
        for n in b_existing:
            if n in pm:
                peer_reads |= (self_attrs(pm[n],ast.Load)-parent_method_names)
        parent_a_established=set()
        for n in a_existing:
            if n in pm: parent_a_established |= self_attrs(pm[n],ast.Store)
        for n in pm:
            if n.startswith("_a_"): parent_a_established |= self_attrs(pm[n],ast.Store)
        established=set()
        for n in a_existing:
            if n in cm: established |= self_attrs(cm[n],ast.Store)
        for n in cm:
            if n.startswith("_a_"): established |= self_attrs(cm[n],ast.Store)
        missing=(peer_reads & parent_a_established)-established-set(class_assigns(cc))
        if missing: fail("A removed peer-visible state assignments: "+",".join(sorted(missing)))

    return cm, cc, ct, c_module_defs, module_renames

def replace_method(cls, name, node):
    for i,n in enumerate(cls.body):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name:
            cls.body[i]=copy.deepcopy(node); return
    cls.body.append(copy.deepcopy(node))

def replace_class_assign(cls, name, node):
    for i,n in enumerate(cls.body):
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(t,ast.Name) and t.id==name for t in targets):
                cls.body[i]=copy.deepcopy(node); return
    cls.body.insert(0,copy.deepcopy(node))

def do_merge(parent,a,b,out,report):
    c=contract(parent)
    am,ac,at,a_mod,a_renames=validate_candidate(parent,a,"A",c)
    bm,bc,bt,b_mod,b_renames=validate_candidate(parent,b,"B",c)
    _,mt,mc,mm=parse(parent)
    parent_assign_before = class_assigns(mc)
    parent_mod = module_defs(mt)

    # Preserve role-owned top-level helper functions/classes. This is required
    # when an owned method instantiates a helper such as _a_EliteArchive.
    top_insert = 0
    if mt.body and isinstance(mt.body[0], ast.Expr) and isinstance(getattr(mt.body[0], "value", None), ast.Constant) and isinstance(mt.body[0].value.value, str):
        top_insert = 1
    while top_insert < len(mt.body) and isinstance(mt.body[top_insert], (ast.Import, ast.ImportFrom)):
        top_insert += 1
    for i,node in enumerate(mt.body):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if node.name.startswith("_a_") and node.name in a_mod:
                mt.body[i]=copy.deepcopy(a_mod[node.name])
            elif node.name.startswith("_b_") and node.name in b_mod:
                mt.body[i]=copy.deepcopy(b_mod[node.name])
    new_top = []
    for name,node in a_mod.items():
        if name not in parent_mod and name.startswith("_a_"):
            new_top.append(copy.deepcopy(node))
    for name,node in b_mod.items():
        if name not in parent_mod and name.startswith("_b_"):
            new_top.append(copy.deepcopy(node))
    mt.body[top_insert:top_insert] = new_top

    # A-owned existing methods and new _a_* helpers.
    for name in c["ownership"]["A"]["existing_methods"]:
        if name in am: replace_method(mc,name,am[name])
    for name,node in am.items():
        if name.startswith("_a_") and name not in mm: replace_method(mc,name,node)

    # A owns module imports: replace parent import nodes with A's import nodes.
    a_imports=imports(at)
    mt.body=[n for n in mt.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    insert_at=0
    if mt.body and isinstance(mt.body[0],ast.Expr) and isinstance(getattr(mt.body[0],"value",None),ast.Constant) and isinstance(mt.body[0].value.value,str):
        insert_at=1
    mt.body[insert_at:insert_at]=[copy.deepcopy(n) for n in a_imports]

    # B-owned existing methods and new _b_* helpers.
    for name in c["ownership"]["B"]["existing_methods"]:
        if name in bm: replace_method(mc,name,bm[name])
    for name,node in bm.items():
        if name.startswith("_b_") and name not in mm: replace_method(mc,name,node)

    b_assign=class_assigns(bc)
    if "batch" in b_assign: replace_class_assign(mc,"batch",b_assign["batch"])

    ast.fix_missing_locations(mt)
    text=ast.unparse(mt).rstrip()+"\n"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(text,encoding="utf-8")
    parse(out)

    rep={
      "status":"merged",
      "ownership":c["ownership"],
      "A_changed":[n for n in c["ownership"]["A"]["existing_methods"] if n in am and n in mm and dump(am[n])!=dump(mm[n])] +
                  [n for n in am if n.startswith("_a_") and n not in mm] +
                  [n for n in a_mod if n.startswith("_a_") and (n not in parent_mod or dump(a_mod[n])!=dump(parent_mod[n]))],
      "B_changed":[n for n in c["ownership"]["B"]["existing_methods"] if n in bm and n in mm and dump(bm[n])!=dump(mm[n])] +
                  [n for n in bm if n.startswith("_b_") and n not in mm] +
                  [n for n in b_mod if n.startswith("_b_") and (n not in parent_mod or dump(b_mod[n])!=dump(parent_mod[n]))],
      "batch_changed": (
          ("batch" in b_assign) != ("batch" in parent_assign_before)
          or ("batch" in b_assign and "batch" in parent_assign_before
              and dump(b_assign["batch"]) != dump(parent_assign_before["batch"]))
      ),
      "namespace_normalization":{"A":a_renames,"B":b_renames},
    }
    report.write_text(json.dumps(rep,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("contract"); p.add_argument("--parent",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("merge")
    for x in ("parent","a","b","out","report"): p.add_argument("--"+x,type=Path,required=True)
    p=sub.add_parser("validate")
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--role",choices=["A","B"],required=True)
    p.add_argument("--contract",type=Path,required=True)
    ns=ap.parse_args()
    if ns.cmd=="contract":
        ns.out.write_text(json.dumps(contract(ns.parent),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    elif ns.cmd=="validate":
        c=json.loads(ns.contract.read_text(encoding="utf-8"))
        validate_candidate(ns.parent,ns.candidate,ns.role,c)
        print("ROLE_VALIDATION_PASS")
    else:
        do_merge(ns.parent,ns.a,ns.b,ns.out,ns.report)

if __name__=="__main__":
    main()
