trainning: playground.ipynb

inference: inference.ipynb

## prerequisite

安装[unidecode](https://github.com/avian2/unidecode)预处理文字，简化token映射逻辑。更改：

```diff
diff --git a/unidecode/__init__.py b/unidecode/__init__.py
index 5633c45..07543e6 100644
--- a/unidecode/__init__.py
+++ b/unidecode/__init__.py
@@ -132,7 +132,8 @@ def _unidecode(string: str, errors: str, replace_str:str) -> str:
                 repl = char
             else:
                 raise UnidecodeError('invalid value for errors parameter %r' % (errors,))
-
+        if repl=="":
+            repl = replace_str
         retval.append(repl)

     return ''.join(retval)
```

```
pip wheel .
```

pip install 本地更改的unidecode