SYSTEM PROMPT — lsFusion TASK RULES

SCOPE: lsFusion

This rule set applies to ALL tasks related to lsFusion
(including analysis, how-to, examples, documentation lookup,
project exploration, and code writing).

These rules MUST be followed.

----------------------------------------------------------------

LSFUSION BRIEF

The assistant MUST keep the relevant lsFusion brief in context
before solving an lsFusion task.

At minimum, the assistant MUST identify which of these groups
are involved:
- modules and namespaces
- classes
- properties
- actions
- forms
- events, constraints, aggregations, integration,
  or other elements

Core element map:
- `CLASS` defines object types, inheritance, polymorphism
- `property` defines declarative logic;
  it may be `DATA` or calculated
- `action` defines imperative logic that changes state
  or calls internal / external logic
- `FORM` defines UI, query, and report structure with
  `OBJECTS`, `PROPERTIES`, `FILTERS`
- `event` reacts to changes, typically with `WHEN`
  or form `ON ...`
- `constraint` defines integrity checks validated on `APPLY`
- `GROUP` / `PARTITION` define aggregation
  and window logic
- `GROUP AGGR` / `AGGR` define aggregate object maintenance
- `MODULE` is the `.lsf` reuse and dependency unit

----------------------------------------------------------------

MANDATORY WORKFLOW

1. ELEMENT IDENTIFICATION ORDER (MANDATORY)
   The assistant MUST reason about lsFusion elements
   strictly in the following order:
   1) element types, modules, classes
   2) properties
   3) actions
   4) forms
   5) other elements

   The assistant MUST NOT jump straight into actions or forms
   before clarifying the module / class / property context.

2. TOOL USAGE (MANDATORY)
   For lsFusion tasks, the assistant MUST actively use
   ALL of the following categories:
   - how-to guidance / examples / analogies
   - documentation lookup
   - structured element search in the project

3. IDE / VALIDATION RULE (MANDATORY)
   If IDE diagnostics or error checking are available,
   the assistant MUST use them.

   Pure syntax validation is acceptable only as a fallback
   when IDE diagnostics or execution checks are unavailable.

----------------------------------------------------------------

RULES FOR USING LSFUSION TOOLS

A. HOW-TO AND EXAMPLES

1. For any code-related lsFusion task, the assistant MUST
   retrieve how-tos or examples first.

2. The assistant MUST decompose the task into small sub-tasks
   that each produce a small amount of code.

3. The assistant SHOULD prefer how-to style reasoning
   over speculative large rewrites.

4. The assistant SHOULD reuse platform patterns from examples
   before inventing custom structure.

B. DOCUMENTATION LOOKUP

1. Before requesting documentation, the assistant MUST first
   determine the current element types.

2. The assistant MUST retrieve definitions and syntax
   for those element types before editing.

3. If syntax, behavior, or capability is uncertain,
   the assistant MUST consult documentation before proceeding.

4. Community retrieval SHOULD be used only when docs
   and how-tos are insufficient for a deep or ambiguous task.

C. ELEMENT SEARCH

1. The assistant MUST prefer structured element search
   over plain text search.

2. Before searching, the assistant MUST determine
   the needed element types, modules, and classes.

3. The assistant MUST try to find the required elements
   in one structured search call with the correct filters.

4. If the target cannot be found, the assistant MUST do
   at least one fallback:
   - minimal-filter search to get a project brief
   - related-element search from already found elements

5. The assistant SHOULD prefer keyword-based search
   over regex when possible.

6. The assistant MUST estimate and set output size
   and timeout intentionally based on task complexity.

----------------------------------------------------------------

CONCURRENT-AGENT RULES

1. OWNERSHIP AND SCOPE
   Before editing, the assistant MUST announce which modules
   or files it is touching.

   The assistant MUST keep scope narrow and prefer one module
   or one coherent feature slice per agent.

   The assistant MUST NOT make opportunistic edits
   outside the claimed scope.

2. COLLISION AVOIDANCE
   The assistant MUST re-read a target file
   immediately before patching it.

   If another agent may be editing the same file or module,
   the assistant MUST stop and reduce scope
   or coordinate through the user.

   The assistant MUST avoid broad reformatting,
   renaming, or file moves unless explicitly requested.

3. PATCH DISCIPLINE
   The assistant MUST make minimal, local changes.

   The assistant MUST preserve unrelated user changes.

   The assistant MUST NOT edit generated outputs
   under `out/production`.

   If source and generated files differ, the assistant MUST treat
   `src/main/lsfusion` as authoritative.

4. VERIFICATION
   After edits, the assistant MUST verify the changed lsFusion area
   with available project or IDE checks.

   If verification is not possible, the assistant MUST say so
   and identify the exact unchecked risk.

----------------------------------------------------------------

PRACTICAL DEFAULT FOR THIS WORKSPACE

When starting work in this repository,
the assistant SHOULD follow this order:

1) inspect `src/main/lsfusion`
2) identify the relevant module and classes first
3) search project elements with structured filters
4) retrieve matching how-to and documentation
   for the element types involved
5) edit only source files, not generated output
6) re-check the target file just before applying a patch
   if multiple agents may be active

----------------------------------------------------------------

SYNTAX RULES

1. Use single `=` as the default equality operator
   in generated lsFusion code.

   `==` is valid syntax, but it SHOULD NOT be the default style
   unless preserving existing code
   or matching an explicit user request.

2. Properties and forms MUST be declared before use.
   The assistant MUST NOT rely on forward use.

----------------------------------------------------------------

BOOLEAN TYPE RULES

1. The assistant MUST use `FALSE` only for `TBOOLEAN`.

2. For `BOOLEAN`, the only values are `TRUE` and `NULL`.

3. For `BOOLEAN`, `NULL` MUST be treated
   as the default value.

----------------------------------------------------------------

PROPERTY RULES

1. The assistant MUST NOT declare a property
   if it is used only once.

   Exception:
   a property may still be declared if it is added to a form.

2. Every property parameter MUST be used in its expression.
   Unused parameters are forbidden.

3. The assistant MUST assume standard `NULL` propagation
   for property expressions:
   if any parameter is `NULL`, the result is `NULL`.

4. The assistant MUST NOT use `GROUP AGGR`
   inside arbitrary expressions.

   `GROUP AGGR` is allowed only in property definitions.

   When reasoning about it, the assistant MUST treat
   `GROUP AGGR` as `GROUP MAX`
   with an additional constraint.

5. The assistant SHOULD avoid unnecessary conditions
   when the language semantics already produce the required result.

6. The assistant MUST NOT create a property whose expression
   is equal to one of its parameters.

7. The assistant MUST NOT create multiple properties
   with identical expressions.

8. If a property is calculated from another property
   but has different parameters, the assistant SHOULD try
   to keep the same property name.

9. To check whether a property is `NULL`,
   the assistant SHOULD use `IF NOT property(...)`.

   To check that it is not `NULL`,
   the assistant SHOULD use `IF property(...)`.

10. The assistant SHOULD specify `CHARWIDTH`
   in the property definition rather than in form design.

   For a simple property composition that only forwards
   another property, the assistant SHOULD NOT repeat
   `CHARWIDTH` on the derived property unless it must differ.

11. For static objects, the assistant MUST NOT use
    `staticCaption` or `staticName` properties.

    The assistant MUST use `caption` and `name` instead.

12. Property names SHOULD be concise
    and avoid unnecessary words.

13. The assistant SHOULD NOT use words in a property name
    that duplicate parameter class names
    unless required for clarity.

14. The assistant SHOULD NOT specify an explicit namespace
    for a property unless necessary.

----------------------------------------------------------------

PROPERTY NAMING POLICY

1. Property names MUST follow lowerCamelCase,
   as in the official lsFusion coding conventions:
   the first word starts with a lowercase letter,
   and each following word starts with a capital letter.

2. For an object's own primitive attributes,
   the assistant MUST prefer the shortest stable business name
   already used in the project.

   Typical base names in the source are:
   `id`, `name`, `fullName`, `number`, `date`, `dateTime`,
   `status`, `type`, `note`, `details`, `price`, `quantity`,
   `amount`, `email`, `phone`, `address`, `city`, `state`,
   `zip`, `index`, `count`, `color`, `readonly`, `archived`.

3. The assistant MUST reuse an existing base property name
   for the same concept across different classes and signatures
   instead of inventing synonyms.

4. The assistant SHOULD NOT include the owner class name
   in a property's own base attribute
   when a generic name is sufficient.

   Prefer:
   `name(Partner)`, `email(Partner)`, `number(Order)`.

   Avoid:
   `partnerName`, `partnerEmail`, `orderNumber`.

5. The assistant MUST NOT add verbs such as
   `get`, `set`, `calc`, or `compute`,
   or filler words such as `value`, `data`, `info`,
   to a property name unless they are part
   of the actual business meaning.

6. Human-readable wording belongs in the caption,
   not in the identifier.

   The assistant SHOULD keep the property name technical
   and reusable even when the caption is long,
   localized, or contains business phrasing.

----------------------------------------------------------------

FORM RULES

1. To place several objects in one table at once,
   the assistant SHOULD combine them into one object group
   using brackets.

2. In a `FORM ... ORDERS` clause, the assistant MUST use
   only form properties that were already added to the form
   via a `PROPERTIES` block.

   In `ORDERS`, the assistant MUST specify either:
   - the form property name with its parameters, if no explicit alias was given
   - the explicit form property alias, if such an alias was specified

   Raw expressions, objects, or properties not added to the form
   MUST NOT be placed into `ORDERS`.

3. The assistant MUST NOT use `INPUT` inside actions
   added directly to a form through `PROPERTIES`,
   unless that action is used in an `ON CHANGE` handler.

   `INPUT` is allowed only in form property change handlers.

4. The assistant MUST NOT display internal object identifiers
   on a form.

   Meaningful properties MUST be shown instead.

5. The assistant MUST NOT add object-valued properties to forms.

   Primitive or derived primitive / text properties
   MUST be exposed instead.

6. The assistant SHOULD specify a `DESIGN`
   for all interactive forms containing more than four properties.

7. Exception:
   for a trivial form with only one or two objects in `GRID` mode
   and no other properties displayed in `PANEL` mode,
   omitting `DESIGN` is acceptable.

8. In `DESIGN`, the assistant SHOULD prefer moving `BOX(...)`
   containers for tables first.

   `GRID(...)` SHOULD be used only when absolutely necessary.

9. If possible, the assistant SHOULD avoid form designs
    with more than two tables stacked vertically
    and more than two tables placed horizontally.

----------------------------------------------------------------

MODULE DESIGN RULES

1. The assistant MUST split lsFusion code into modules
   by domain logic or feature area,
   not by arbitrary technical grouping.

2. The assistant SHOULD prefer relatively short modules.

   A single broad module SHOULD NOT keep growing
   when the logic naturally separates
   into smaller cohesive modules.

3. The assistant MUST apply low coupling and high cohesion:
   closely related classes, properties, actions, and forms
   SHOULD stay together,
   and cross-module dependencies SHOULD remain narrow and explicit.

4. Module `NAMESPACE` SHOULD be chosen by shared business domain,
   not by the full module name.

5. When a module belongs to an existing domain family,
   the assistant SHOULD reuse that family namespace
   for all its elements.

   A new namespace SHOULD be created only
   for a genuinely new domain,
   not for each technical submodule.

6. If the module name already equals the intended domain namespace,
   omitting `NAMESPACE` is acceptable
   because lsFusion will use the module name as the default.

   Otherwise, the assistant SHOULD specify `NAMESPACE` explicitly.

7. The assistant SHOULD use `REQUIRE`, `EXTEND`,
   abstract properties / actions,
   and form extensions to connect modules
   instead of duplicating logic
   or creating a god module.

8. Before adding code to an existing module,
   the assistant MUST check whether the logic belongs
   to that module's domain.

   If not, the assistant SHOULD create
   or extend a more appropriate module.

9. When introducing a new module,
   the assistant MUST choose dependencies deliberately
   and avoid circular or unnecessary dependencies.

----------------------------------------------------------------

CHANGE SESSION RULES (`NEWSESSION`, `NESTEDSESSION`, `APPLY`)

1. Before introducing `NEWSESSION`, the assistant MUST decide
   which session behavior is required:
   - isolated independent unit -> `NEWSESSION`
   - isolated unit that must also see selected local properties
     from the upper session -> `NEWSESSION NESTED (...)`
   - isolated unit that must see all local properties
     from the upper session -> `NEWSESSION NESTED LOCAL`
   - child dialog or editor that must work with unsaved upper-session
     objects and return its changes to that upper session
     -> `NESTEDSESSION`

2. For actions added to forms,
   there are two main patterns:

   - readonly form pattern:
     the form is effectively browse-only, so actions added to it
     SHOULD run in a new session by default
   - editable form pattern:
     the form has editable properties, so any action added to it
     that uses `NEWSESSION` MUST either:
     `APPLY;`
     `IF canceled() THEN RETURN;`
     before `NEWSESSION`, or be fully independent
     from unsaved changes in that form

3. Plain `NEWSESSION` is the default
   for isolated work that must not accidentally apply
   the caller's pending form changes.

   Typical patterns in the source:
   - readonly list forms with
     `PROPERTIES(...) NEWSESSION NEW, EDIT, DELETE`
   - status transitions or dependent document creation
     after a preceding `APPLY`
   - external or integration actions that isolate HTTP calls
     and persist their own results
   - small immediate UI updates with
     `NEWSESSION { APPLY { ... } }`

4. If inner logic depends on upper-session local state
   such as selections, marks, or import buffers,
   the assistant MUST carry that state explicitly
   through `NESTED (...)` or `NESTED LOCAL`.

5. When using `NEWSESSION NESTED (...)` or
   `NEWSESSION NESTED LOCAL`, the assistant SHOULD preserve
   the same nested local properties on `APPLY`
   if the result must be copied back to the upper session,
   for example with `APPLY NESTED (...)`
   or `APPLY NESTED LOCAL`.

6. The assistant MUST NOT replace `NESTEDSESSION`
   with plain `NEWSESSION` for child forms or dialogs
   attached to a parent object that may still be unsaved
   in the current form session.

7. Before opening a fresh `NEWSESSION` from an action
   started on an edit form, the assistant SHOULD decide
   whether current form changes must be saved first.

   The common pattern is:
   `APPLY;`
   `IF canceled() THEN RETURN;`
   `NEWSESSION { ... }`

   This pattern is used before status changes,
   document generation, and other isolated follow-up actions.

8. After `APPLY`, if later logic depends on whether the save
   succeeded, the assistant MUST check `canceled()`.
   If the failure must be surfaced to the user or integration,
   the assistant SHOULD use `applyMessage()`.

9. The assistant SHOULD keep `NEWSESSION` blocks small
   and purpose-specific: isolate one unit of work,
   apply it if needed, and exit.

   The assistant MUST NOT introduce `NEWSESSION`
   merely to hide session-visibility bugs.
   If upper-session changes must remain visible,
   nested session semantics are required.

----------------------------------------------------------------

IMPORT RULES (`IMPORT`)

1. Before working with `IMPORT`, the assistant MUST identify
   elements in this order:
   - module and namespace that own the import flow
   - target classes that will be created or updated
   - staging properties used during import
   - import actions
   - import forms, if the payload is hierarchical

2. The assistant MUST choose the import style intentionally:
   - flat files (`CSV`, `XLS`, `DBF`, `TABLE`)
     -> prefer `IMPORT ... TO` or `FIELDS`
   - nested `JSON` / `XML`, parent-child structures,
     namespaces, or `EXTID` mapping
     -> prefer form import
   - row-at-a-time integration responses
     -> prefer `FIELDS ... DO`

3. For flat imports that need validation, deduplication,
   multi-pass processing, or post-processing,
   the assistant SHOULD stage data into `LOCAL` properties
   first, usually by `INTEGER` row,
   then process it in a separate
   `FOR imported(INTEGER i)` pass.

4. The assistant SHOULD use `FIELDS ... DO`
   when imported values are consumed only once
   and introducing reusable local properties
   would add noise.

5. The assistant SHOULD specify column mappings explicitly
   when the external template is fixed or sparse.

   Sequential mapping without explicit column IDs
   is acceptable only when column order itself
   is the agreed interface.

6. For form import, the assistant MUST declare
   a dedicated import form before use.

   The form MUST use one object per object group
   with numeric or concrete user classes.

   The form SHOULD mirror the external structure with:
   - `FILTERS` for parent-child links
   - `EXTID`, `FORMEXTID`, groups, and `ATTR`
     only where the external schema requires them

   The assistant MUST remember that importing into a form
   cancels pending changes to imported form properties
   in the current session.

7. The assistant MUST choose format options explicitly
   when the external contract depends on them:
   - `HEADER` / `NOHEADER`
   - `SHEET`
   - `CHARSET`

   The assistant SHOULD prefer `HEADER`
   for stable `CSV` / `XLS` templates,
   because `NOHEADER` can silently map missing
   or mistyped columns to `NULL`.

8. The assistant MUST validate referenced business keys
   before creating or updating persistent objects.

   Typical keys in this project are `id`, `number`,
   partner or item codes, and external references.

   Each reference MUST be checked
   in a separate `FOR`
   using `GROUP SUM 1 BY`
   over the imported key values.

   If possible, the assistant SHOULD NOT write
   resolved references to a separate `LOCAL`
   before the main import logic.

   Missing master data or malformed payloads
   MUST stop the import or surface a clear error.

9. The assistant SHOULD separate raw import
   from domain resolution:
   - first parse the file or payload
     into locals or an import form
   - then check references such as
     item, partner, status, type, or other lookups
   - only then create or update domain objects

10. For user-started batch imports and external integrations,
    the assistant SHOULD isolate persistence in `NEWSESSION`.

    After domain writes, the assistant SHOULD `APPLY;`.
    If later logic depends on success,
    the assistant MUST check `canceled()`
    and surface `applyMessage()` or `throwException(...)`.

    If the import must see upper-session local buffers,
    the assistant MUST use nested session semantics
    instead of plain `NEWSESSION`.

11. The assistant MUST NOT partially persist
    a failed import silently.

    It SHOULD use `MESSAGE`, `RETURN`,
    `throwException`, or an explicit failure flag,
    consistent with the caller:
    - interactive import -> `MESSAGE`
    - API or background integration
      -> exception or explicit failure state

12. When importing booleans,
    the assistant MUST remember workspace boolean rules:
    - `BOOLEAN` uses `TRUE` and `NULL`
    - `FALSE` is valid only for `TBOOLEAN`

13. For create-or-update synchronization imports,
    the assistant MUST separate object creation
    from property updates.

    The assistant MUST use one separate `FOR`
    to create missing objects only.

    If imported key values may be non-unique,
    the creation pass SHOULD iterate by grouped keys
    using `GROUP SUM ... BY`
    rather than by raw imported rows.

    The assistant MUST then use a second separate `FOR`
    to update properties of matched objects.

    The assistant MUST NOT mix object creation
    and property updates in the same loop
    for synchronization imports.

    If full synchronization is required,
    the assistant SHOULD add an explicit delete step.

14. If `LOCAL` staging properties
    are used only in one import action,
    the assistant MUST declare them
    inside that action.

    The assistant SHOULD NOT lift such `LOCAL` properties
    to module scope without need.

    Exception:
    a `LOCAL` property may be declared outside the action
    only when it must be used by an import form
    or reused by several related actions.
