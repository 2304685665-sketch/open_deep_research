"""T1 SQL 攻击/正常/资源耗尽用例定义——纯数据，不包含任何执行逻辑。

由 eval_sql_guard.py 读取，分别对新旧两个版本的 sql_executor 各跑一遍。
每条用例是一个 dict：id / sql / category / expect / note。

expect 取值仅用于人工标注"这条用例设计意图上预期是攻击还是正常查询"，
不用于自动判定结果——真正的判定（拦截/成功/误拦/基础设施失败）由
eval_sql_guard.py 基于实际运行产物（SHA-256、integrity_check、新文件、
返回内容）独立算出，不依赖这个标注。
"""

ATTACK_CASES = [
    {"id": "A01", "sql": "INSERT INTO companies(name, industry, founded_year) VALUES ('Evil Corp', 'hacking', 2026)",
     "category": "经典写操作", "note": "最基础的INSERT，验证关键字检查本身的最低门槛"},
    {"id": "A02", "sql": "UPDATE companies SET name = 'HACKED' WHERE id = 1",
     "category": "经典写操作", "note": "经典UPDATE"},
    {"id": "A03", "sql": "DELETE FROM companies WHERE id = 1",
     "category": "经典写操作", "note": "经典DELETE"},
    {"id": "A04", "sql": "DROP TABLE companies",
     "category": "经典写操作", "note": "经典DROP TABLE"},
    {"id": "A05", "sql": "-- just a comment\nINSERT INTO companies(name, industry, founded_year) VALUES ('Evil', 'x', 2026)",
     "category": "注释前缀绕过", "note": "行注释前缀隐藏INSERT"},
    {"id": "A06", "sql": "/* block comment */ATTACH DATABASE 'evil_a06.db' AS evil",
     "category": "注释前缀绕过", "note": "块注释前缀隐藏ATTACH"},
    {"id": "A07", "sql": "AtTaCh DATABASE 'evil_a07.db' AS evil",
     "category": "大小写混淆", "note": "大小写混写的ATTACH，测试.lower()归一化是否覆盖关键字判断本身"},
    {"id": "A08", "sql": "DrOp TaBlE companies",
     "category": "大小写混淆", "note": "大小写混写的DROP"},
    {"id": "A09", "sql": "   \n\t  DELETE FROM companies",
     "category": "空白混淆", "note": "前导空白+制表符+换行混合，测试.strip()是否被绕过"},
    {"id": "A10", "sql": "WITH c AS (SELECT 1) DELETE FROM companies",
     "category": "WITH包装写操作", "note": "WITH子句包装的DELETE——SQLite允许WITH前缀写操作，不限于SELECT"},
    {"id": "A11", "sql": "WITH c AS (SELECT 1) INSERT INTO companies(name, industry, founded_year) VALUES ('Evil', 'x', 2026)",
     "category": "WITH包装写操作", "note": "WITH子句包装的INSERT"},
    {"id": "A12", "sql": "WITH c AS (SELECT 1) UPDATE companies SET name = 'HACKED'",
     "category": "WITH包装写操作", "note": "WITH子句包装的UPDATE"},
    {"id": "A13", "sql": "SELECT * FROM companies; DROP TABLE companies;",
     "category": "多语句", "note": "SELECT后接DROP——经验证Python sqlite3.execute()本身拒绝多语句，这条用来确认该保护在新旧版本下是否都生效（如果生效，预期两版本都被拦截，这是stdlib层面的保护，不是sql_executor自己的功劳）"},
    {"id": "A14", "sql": "SELECT 1; DELETE FROM companies;",
     "category": "多语句", "note": "多语句变体"},
    {"id": "A15", "sql": "PRAGMA journal_mode=DELETE",
     "category": "PRAGMA写", "pragma_probe": "journal_mode", "pragma_target": "delete",
     "note": "修改journal_mode为DELETE——但demo.db本来就是DELETE模式（已实测确认: "
             "PRAGMA journal_mode 查询返回 'delete'），这是已知的无操作(no-op)用例，"
             "不是有效的拦截测试，保留只是为了和A23(WAL)对比"},
    {"id": "A16", "sql": "PRAGMA user_version=999",
     "category": "PRAGMA写", "pragma_probe": "user_version", "pragma_target": 999,
     "note": "修改user_version，直接写入数据库文件头，demo.db原值是0（已实测确认），"
             "目标值999，是真正有区分度的测试"},
    {"id": "A17", "sql": "PRAGMA writable_schema=ON",
     "category": "PRAGMA写", "pragma_probe": None, "pragma_target": None,
     "note": "writable_schema是连接级设置，每个新连接默认关闭，我们的测试方法论"
             "（单语句单连接即关闭）结构上无法验证这条PRAGMA在同一连接后续操作中"
             "是否真的生效——这不是攻击被拦截，是测试方法本身覆盖不到这个场景"},
    {"id": "A23", "sql": "PRAGMA journal_mode=WAL",
     "category": "PRAGMA写", "pragma_probe": "journal_mode", "pragma_target": "wal",
     "note": "修改journal_mode为WAL——demo.db原值是delete，目标值wal不同于原值，"
             "是比A15更有区分度的测试；WAL模式切换成功通常会在目录里出现"
             "demo.db-wal/demo.db-shm伴生文件，需要和SQLite正常产生的辅助文件"
             "区分开，不能把这类文件误判成攻击新建的可疑文件"},
    {"id": "A18", "sql": "ATTACH DATABASE 'evil_a18.db' AS evil",
     "category": "ATTACH/DETACH", "note": "经典ATTACH，在磁盘上创建新数据库文件"},
    {"id": "A19", "sql": "ATTACH 'evil_a19.db' AS evil2",
     "category": "ATTACH/DETACH", "note": "ATTACH变体语法（省略DATABASE关键字）"},
    {"id": "A20", "sql": "DETACH DATABASE main",
     "category": "ATTACH/DETACH", "note": "DETACH主数据库，异常场景"},
    {"id": "A21", "sql": "VACUUM INTO 'evil_a21_vacuum.db'",
     "category": "VACUUM INTO写文件", "note": "VACUUM INTO会把整个数据库内容写入一个新文件"},
    {"id": "A22", "sql": "SELECT load_extension('/tmp/does_not_exist_evil.so')",
     "category": "load_extension", "note": "尝试加载原生扩展——即使目标文件不存在，关键是看load_extension这个调用本身是否被允许尝试（是否需要先enable_load_extension，默认应该是禁用的）"},
]

NORMAL_CASES = [
    {"id": "N01", "sql": "SELECT * FROM companies", "note": "基础SELECT"},
    {"id": "N02", "sql": "SELECT name, industry FROM companies WHERE founded_year > 2020", "note": "带WHERE条件"},
    {"id": "N03", "sql": "SELECT COUNT(*) FROM funding_rounds", "note": "聚合COUNT"},
    {"id": "N04", "sql": "SELECT AVG(amount_usd) FROM funding_rounds", "note": "聚合AVG"},
    {"id": "N05", "sql": "SELECT c.name, f.amount_usd FROM companies c JOIN funding_rounds f ON c.id = f.company_id", "note": "JOIN"},
    {"id": "N06", "sql": "SELECT name FROM companies ORDER BY founded_year DESC LIMIT 3", "note": "ORDER BY + LIMIT"},
    {"id": "N07", "sql": "SELECT industry, COUNT(*) FROM companies GROUP BY industry", "note": "GROUP BY"},
    {"id": "N08", "sql": "SELECT * FROM companies WHERE id IN (SELECT company_id FROM funding_rounds WHERE amount_usd > 20000000)", "note": "只读子查询"},
    {"id": "N09", "sql": "SELECT name, sql FROM sqlite_master WHERE type='table'", "note": "schema内省，sql_executor设计上依赖这条路径"},
    {"id": "N10", "sql": "  \n  SELECT 1", "note": "合法查询带前导空白，验证不会被误拦"},
    {"id": "N11", "sql": "-- 这是一条注释\nSELECT * FROM companies LIMIT 1", "note": "合法查询带注释前缀，验证注释剥离不会误伤正常查询"},
    {"id": "N12", "sql": "SeLeCt * FROM companies LIMIT 1", "note": "合法查询大小写混写，验证大小写归一化不会误伤"},
]

RESOURCE_CASES = [
    {"id": "R01", "sql": "SELECT * FROM companies a, companies b, companies c, companies d, companies e, companies f, companies g, companies h",
     "note": "8路自连接笛卡尔积（4行^8条 ≈ 65536行的笛卡尔积基数，用来验证运行时限制/结果集截断是否生效，不是解析器层面的限制）"},
    {"id": "R02", "sql": "SELECT * FROM companies a, companies b, companies c, companies d, companies e, companies f, companies g, companies h, companies i, companies j, companies k, companies l",
     "note": "12路自连接（4^12≈1670万行基数），比R01更大一个量级，用来模拟'深度递归/深度计算膨胀'的效果。"
             "说明：原计划里的'深度递归CTE'字面意义上无法用'以SELECT开头'的方式构造——"
             "SQLite的WITH RECURSIVE语法本身要求WITH作为语句的第一个关键字，"
             "而WITH前缀的查询在当前sql_executor的分类器里会被直接拒绝（归为'with'，不进入运行时），"
             "所以用更大规模的笛卡尔积代替，同样达到'真正考验运行时限制'这个目的，"
             "但测试的是运行时资源限制而不是递归本身——如果这个替代理解不符合预期，请明确指出，我会改用别的构造方式。"},
]
