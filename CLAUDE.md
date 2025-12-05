# CLAUDE.md - AI Agent Configuration for Maximum Capability

## ⚠️ FIRST: Code Quality Setup Check

**Before ANY development work, run this check:**

```bash
[ -f ".code-quality-setup" ] || ~/mcp-config/setup-code-quality.sh
```

| File Exists?                  | Action                                   |
| ----------------------------- | ---------------------------------------- |
| `.code-quality-setup` present | ✅ Proceed with development              |
| `.code-quality-setup` missing | Run `~/mcp-config/setup-code-quality.sh` |

---

## 🚀 MCP Server Reference (12 Servers)

### Server Selection Matrix

| Task                            | Primary Server        | Fallback     | Why                                 |
| ------------------------------- | --------------------- | ------------ | ----------------------------------- |
| **Find code by meaning**        | `code-index`          | `filesystem` | Semantic search = 100x fewer tokens |
| **Store knowledge permanently** | `qdrant`              | `memory`     | Vector DB persists across sessions  |
| **Project context**             | `memory-bank`         | `memory`     | Project-specific, structured        |
| **Complex reasoning**           | `sequential-thinking` | -            | Breaks down multi-step problems     |
| **Read/write files**            | `filesystem`          | -            | Direct file operations              |
| **Git operations**              | `git`                 | bash         | Native git commands                 |
| **GitHub PRs/issues**           | `github`              | fetch        | API access, not scraping            |
| **Library docs**                | `context7`            | fetch        | Always current, structured          |
| **Web content**                 | `fetch`               | playwright   | Lightweight, fast                   |
| **Browser automation**          | `playwright`          | -            | JS rendering, screenshots           |
| **Kubernetes**                  | `kubernetes`          | bash         | Native K8s API                      |

---

## 🧠 Memory Architecture (4 Servers)

### When to Use Each Memory Server

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  QDRANT (Vector Database) ─────────────────────────────────│
│  └── Permanent semantic storage                             │
│  └── Survives restarts, persists forever                    │
│  └── USE FOR: Architecture decisions, patterns, learnings   │
│  └── QUERY: "Find similar code patterns to authentication"  │
│                                                             │
│  MEMORY (Knowledge Graph) ─────────────────────────────────│
│  └── Entities and relationships                             │
│  └── User preferences, project facts                        │
│  └── USE FOR: "User prefers Python", "Project uses FastAPI" │
│                                                             │
│  MEMORY-BANK (Project Context) ────────────────────────────│
│  └── Current project state and progress                     │
│  └── Task tracking, decisions made                          │
│  └── USE FOR: "Currently implementing auth module"          │
│                                                             │
│  SEQUENTIAL-THINKING (Reasoning) ──────────────────────────│
│  └── Complex multi-step problem solving                     │
│  └── Break down large tasks                                 │
│  └── USE FOR: Planning migrations, architecture changes     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Memory Server Commands

| Server                  | Key Operations                                     | Example                               |
| ----------------------- | -------------------------------------------------- | ------------------------------------- |
| **qdrant**              | `store`, `find`, `search`                          | Store architectural pattern for reuse |
| **memory**              | `create_entities`, `search_nodes`, `add_relations` | "Daniel prefers Python for backend"   |
| **memory-bank**         | `initialize_memory_bank`, `update_context`         | Track current task progress           |
| **sequential-thinking** | `create_thinking_session`, `add_thought`           | Plan complex refactoring              |

---

## 📁 Code Operations (3 Servers)

### Token Efficiency for Large Codebases

**CRITICAL: For codebases with 100M+ lines, NEVER read entire files. Use semantic search.**

| Approach                     | Tokens Used  | Speed                     |
| ---------------------------- | ------------ | ------------------------- |
| Read entire file             | 50,000+      | Slow, fills context       |
| Grep search                  | 5,000-20,000 | Medium                    |
| `code-index` semantic search | 200-500      | Fast, precise             |
| `filesystem` targeted read   | 500-2,000    | Fast if you know location |

### Code Server Selection

```
┌─────────────────────────────────────────────────────────────┐
│                 CODE OPERATION FLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  "Find authentication logic"                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                           │
│  │ code-index  │ ← FIRST: Semantic search                  │
│  └─────────────┘   Returns: file paths + relevant snippets │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                           │
│  │ filesystem  │ ← THEN: Read specific sections            │
│  └─────────────┘   Only read what you need                 │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                           │
│  │    git      │ ← FINALLY: Commit changes                 │
│  └─────────────┘   Use conventional commits                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Server Commands

| Server         | Operations                                  | Best For                              |
| -------------- | ------------------------------------------- | ------------------------------------- |
| **code-index** | `search`, `index`, `find_similar`           | Finding code by meaning, not keywords |
| **filesystem** | `read_file`, `write_file`, `list_directory` | Direct file operations                |
| **git**        | `status`, `commit`, `diff`, `log`, `branch` | Version control                       |

---

## 🌐 Web & Documentation (3 Servers)

### Server Selection

| Need                   | Server       | Why                        |
| ---------------------- | ------------ | -------------------------- |
| Library API docs       | `context7`   | Structured, always current |
| Any web page           | `fetch`      | Fast, returns markdown     |
| JavaScript-heavy sites | `playwright` | Full browser rendering     |
| Screenshots needed     | `playwright` | Can capture visuals        |

### Usage Patterns

```bash
# Get library documentation (PREFER THIS)
context7: search_docs("fastapi authentication")

# Fetch web content
fetch: fetch_url("https://example.com/api/docs")

# Browser automation (when JS rendering needed)
playwright: navigate("https://app.example.com")
playwright: screenshot()
```

---

## 🏗️ DevOps (2 Servers)

### GitHub Server (`github`)

| Operation   | Command                           | Use Case                      |
| ----------- | --------------------------------- | ----------------------------- |
| Create PR   | `create_pull_request`             | Submit changes for review     |
| List issues | `list_issues`                     | Find bugs/features to work on |
| Review PR   | `get_pull_request`, `add_comment` | Code review                   |
| Search code | `search_code`                     | Find patterns across repos    |
| Get file    | `get_file_contents`               | Read from any branch          |

### Kubernetes Server (`kubernetes`)

| Operation         | Command        | Use Case                 |
| ----------------- | -------------- | ------------------------ |
| Get pods          | `get_pods`     | Check running containers |
| Get logs          | `get_logs`     | Debug issues             |
| Apply manifest    | `apply`        | Deploy changes           |
| Describe resource | `describe`     | Troubleshoot             |
| Get services      | `get_services` | Find endpoints           |

---

## 📋 Mandatory Workflows

### Session Start Checklist

```
1. [ -f ".code-quality-setup" ] || ~/mcp-config/setup-code-quality.sh
2. memory: Check for user context and preferences
3. memory-bank: Load current project status
4. qdrant: Search for relevant stored patterns
5. git: Check current branch and status
```

### Before Writing Code

```
1. code-index: Search for existing similar code (DON'T DUPLICATE)
2. context7: Get latest library documentation
3. memory-bank: Review current task context
4. sequential-thinking: Plan approach for complex tasks
```

### After Writing Code

```
1. pre-commit run --all-files (auto on commit)
2. qdrant: Store important patterns/decisions
3. memory-bank: Update progress
4. git: Commit with conventional message
5. github: Create PR if ready
```

### For Large Codebase Operations

```
NEVER:
- Read entire files to "understand" them
- Load multiple large files into context
- Search by reading directory contents

ALWAYS:
- Use code-index for semantic search first
- Read only the specific functions/classes needed
- Use qdrant to store patterns for reuse
- Let code-index find related code
```

---

## 🎯 Task-Specific Server Selection

### "Fix a bug"

```
1. code-index → Find relevant code
2. filesystem → Read specific section
3. context7 → Check library docs if needed
4. filesystem → Write fix
5. git → Commit
```

### "Add new feature"

```
1. memory-bank → Check project context
2. code-index → Find similar existing features
3. sequential-thinking → Plan implementation
4. context7 → Get library docs
5. filesystem → Implement
6. git → Commit
7. github → Create PR
```

### "Understand large codebase"

```
1. code-index → Search by concepts, not files
2. qdrant → Store understanding as you learn
3. memory-bank → Track what you've explored
4. sequential-thinking → Build mental model
```

### "Debug production issue"

```
1. kubernetes → Get pod logs
2. code-index → Find relevant code
3. github → Check recent changes
4. sequential-thinking → Analyze root cause
5. memory → Store resolution for future
```

### "Code review"

```
1. github → Get PR details
2. code-index → Find related code patterns
3. qdrant → Check stored best practices
4. github → Add review comments
```

---

## ⚡ Token Efficiency Guide

### For 100M+ Line Codebases

| Operation             | Tokens | Method                                 |
| --------------------- | ------ | -------------------------------------- |
| Find auth code        | 300    | `code-index search "authentication"`   |
| Read entire auth.py   | 50,000 | ❌ NEVER DO THIS                       |
| Read auth function    | 500    | `filesystem read_file` with line range |
| Find similar patterns | 400    | `code-index find_similar`              |
| Store pattern         | 100    | `qdrant store`                         |
| Recall pattern        | 200    | `qdrant search`                        |

### Context Management Rules

1. **Never exceed 50% context with file contents**
2. **Use code-index search, not file reading**
3. **Store findings in qdrant, don't re-search**
4. **Read specific line ranges, not entire files**
5. **Use memory-bank to track progress between turns**

---

## 🔧 Code Quality Enforcement

### Pre-commit Checks (Automatic)

| Check    | Tool           | Blocks Commit If      |
| -------- | -------------- | --------------------- |
| Format   | ruff-format    | Code not formatted    |
| Lint     | ruff           | Errors or warnings    |
| Types    | mypy           | Type mismatches       |
| Security | bandit         | Vulnerabilities found |
| Secrets  | detect-secrets | API keys in code      |
| Tests    | pytest         | Coverage < 80%        |
| Commits  | commitizen     | Bad commit message    |

### Commit Message Format

```
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
```

### If Commit Fails

```bash
# See failures
pre-commit run --all-files

# Auto-fix
ruff format . && ruff check --fix .

# Skip (emergency only!)
git commit --no-verify -m "message"
```

---

## 📊 MCP Server Health Check

In Claude Code, type `/mcp` to verify all 12 servers:

```
✓ memory              - Knowledge graph
✓ memory-bank         - Project context
✓ sequential-thinking - Complex reasoning
✓ code-index          - Semantic search
✓ qdrant              - Vector database
✓ filesystem          - File operations
✓ git                 - Version control
✓ github              - GitHub API
✓ kubernetes          - K8s management
✓ fetch               - Web content
✓ context7            - Library docs
✓ playwright          - Browser automation
```

---

## 🎓 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              MCP SERVER QUICK REFERENCE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FIND CODE      → code-index (semantic) → filesystem (read)│
│  STORE MEMORY   → qdrant (permanent) / memory (graph)      │
│  PROJECT STATE  → memory-bank                               │
│  COMPLEX TASK   → sequential-thinking                       │
│  LIBRARY DOCS   → context7                                  │
│  WEB CONTENT    → fetch (fast) / playwright (JS)           │
│  GIT OPERATIONS → git                                       │
│  GITHUB API     → github                                    │
│  KUBERNETES     → kubernetes                                │
│  FILES          → filesystem                                │
│                                                             │
│  BEFORE CODE:   [ -f ".code-quality-setup" ] || setup      │
│  AFTER CODE:    pre-commit runs automatically               │
│  COMMIT FORMAT: type(scope): description                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Principles

1. **Search semantically, don't read blindly** - Use code-index first
2. **Store knowledge permanently** - Use qdrant for patterns
3. **Track context** - Use memory-bank for project state
4. **Plan complex work** - Use sequential-thinking
5. **Check quality first** - Verify .code-quality-setup exists
6. **Commit properly** - Conventional commits, run pre-commit
7. **Use the right server** - Refer to selection matrix above
