# Cognitive Memory for LLM Agents

A persistent, evidence-backed, hierarchical cognitive memory system designed to give LLM agents long-term memory, temporal awareness, relational understanding, and the ability to continuously consolidate and evolve what they have learned.


======================================================================

OVERVIEW
======================================================================

Large Language Models are powerful at reasoning within the context they
are given, but a context window is not the same thing as memory.

Traditional LLM applications commonly use vector databases and RAG to
retrieve information. While this is useful, long-term cognitive memory
requires more than retrieving semantically similar text.

A useful memory system should be able to understand:

    - What information is worth remembering
    - Where a memory came from
    - Whether information is still valid
    - How memories relate to each other
    - How information changes over time
    - Whether two memories represent the same fact
    - Whether two memories contradict each other
    - Which memories are important or repeatedly reinforced
    - Which memories should become historical or inactive
    - How memories should be consolidated as new experiences arrive
    - How to retrieve the right information without overwhelming
      the LLM's context window

This project explores the design and implementation of a cognitive
memory architecture for LLM agents.

The system combines:

    Canonical Semantic Memory
            +
    Evidence & Provenance
            +
    Vector Retrieval
            +
    Graph Relationships
            +
    Temporal Reasoning
            +
    Hierarchical Memory
            +
    Asynchronous Dreaming
            +
    Memory Evolution
            +
    Cost-Aware Retrieval


======================================================================
CORE IDEA
======================================================================

The system is built around two interconnected loops.

ONLINE MEMORY LOOP
------------------

The online loop handles the agent's normal interaction with the world.

    EXPERIENCE
        |
        v
    MEMORY FORMATION
        |
        v
    CANONICAL MEMORY
        |
        +-------------------+
        |                   |
        v                   v
    VECTOR MEMORY       GRAPH MEMORY
        |                   |
        +---------+---------+
                  |
                  v
          HYBRID RETRIEVAL
                  |
                  v
          TEMPORAL REASONING
                  |
                  v
           CONTEXT BUILDER
                  |
                  v
              LLM AGENT
                  |
                  v
          NEW EXPERIENCE
                  |
                  +----------------------+
                                         |
                                         v
                                   DREAM ENGINE


OFFLINE MEMORY EVOLUTION LOOP
-----------------------------

The Dream Engine asynchronously examines existing memories and improves
their organization.

    EXISTING MEMORIES
           |
           v
      DREAM ENGINE
           |
           v
    CANDIDATE SELECTION
           |
           v
       NEIGHBORHOOD
           |
           v
       CLUSTERING
           |
           v
    STRUCTURAL ANALYSIS
           |
           +-----------------------------+
           |             |               |
           v             v               v
       DUPLICATES     CONFLICTS     RELATIONSHIPS
           |             |               |
           +-------------+---------------+
                         |
                         v
                 TEMPORAL REASONING
                         |
                         v
                    JEV DECISION
                         |
                         v
                  MEMORY MANAGER
                         |
                         v
                  CANONICAL MEMORY
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
         OBJECT       VECTOR       GRAPH
          STORE       STORE        STORE


======================================================================
HIGH-LEVEL ARCHITECTURE
======================================================================


                         USER / ENVIRONMENT
                                |
                                v
                         USER EXPERIENCE
                                |
                                v
                           LLM AGENT
                                |
                                v
                     +---------------------+
                     |    MEMORY API /     |
                     |    AGENT TOOLS      |
                     +---------------------+
                                |
                                v
                     +---------------------+
                     | MEMORY ORCHESTRATOR |
                     +---------------------+
                          /     |      \
                         /      |       \
                        v       v        v
               MEMORY FORMATION  RETRIEVAL  DREAM ENGINE
                        |       |        |
                        v       |        |
                    JEV /       |        |
                DECISION LAYER  |        |
                        |       |        |
                        v       |        v
                 MEMORY MANAGER |   MEMORY EVOLUTION
                        |       |        |
                        +-------+--------+
                                |
                                v
                    +-----------------------+
                    |   CANONICAL MEMORY    |
                    |     MEMORY KERNEL     |
                    +-----------------------+
                         /       |        \
                        /        |         \
                       v         v          v
              +----------+  +--------+  +--------+
              | EVIDENCE |  | QDRANT |  | NEO4J  |
              | / OBJECT |  | VECTOR |  | GRAPH  |
              |  STORE   |  | MEMORY |  | MEMORY |
              +----------+  +--------+  +--------+
                       \         |          /
                        \        |         /
                         +-------+--------+
                                 |
                                 v
                        HYBRID RETRIEVAL
                                 |
                                 v
                        TEMPORAL REASONING
                                 |
                                 v
                         CONTEXT BUILDER
                                 |
                                 v
                             LLM AGENT
                                 |
                                 v
                          NEW EXPERIENCE
                                 |
                                 +------------------+
                                                    |
                                                    v
                                             DREAM ENGINE
                                                    |
                                                    v
                                           MEMORY EVOLUTION


======================================================================
THE CENTRAL ARCHITECTURAL PRINCIPLE
======================================================================

The most important architectural decision is that Canonical Memory is
the semantic source of truth.

The Object Store, Vector Store, and Graph Store are representations,
indexes, and retrieval mechanisms derived from canonical memory.

They are NOT independent sources of truth.

                         CANONICAL MEMORY
                         /       |       \
                        /        |        \
                       v         v         v
                  EVIDENCE     VECTOR     GRAPH
                  / FILES      INDEX      INDEX

This means that the system should be able to rebuild its vector and
graph representations from canonical memory without losing the
underlying semantic state.


======================================================================
MAJOR ARCHITECTURE COMPONENTS
======================================================================


1. MEMORY KERNEL
----------------

The Memory Kernel defines what a memory actually is.

It provides the semantic foundation for the complete system, including:

    - Memory
    - Episode
    - Message
    - Evidence
    - Entity
    - Relationship
    - Provenance
    - Temporal state
    - Version
    - Lifecycle
    - Confidence
    - Importance
    - Reinforcement
    - Access information

The kernel establishes stable identities and the rules that every other
component must follow.


2. CANONICAL MEMORY
-------------------

Canonical Memory is the semantic source of truth.

It represents the current known state of the agent's long-term memory.

Other storage systems represent or index this state.

The relationship is:

    EXPERIENCE
        |
        v
    CANONICAL MEMORY
        |
        +------> OBJECT / EVIDENCE STORE
        |
        +------> VECTOR REPRESENTATION
        |
        +------> GRAPH REPRESENTATION

The canonical layer is therefore independent from the specific database
technology used for retrieval.


3. MEMORY FORMATION
-------------------

Memory Formation determines how raw experiences become durable memories.

The general pipeline is:

    EXPERIENCE
        |
        v
    INGESTION
        |
        v
    MEMORY EVENT
        |
        v
    EPISODE / MESSAGE
        |
        v
    WORKING MEMORY
        |
        v
    MEMORY FORMATION
        |
        v
    ENTITY / TEMPORAL / TYPE ANALYSIS
        |
        v
    CANDIDATE MEMORY
        |
        v
    EXISTING MEMORY MATCHING
        |
        v
    JEV DECISION
        |
        v
    MEMORY MANAGER
        |
        v
    CANONICAL MEMORY

Not every statement becomes a permanent memory.

The system selectively determines whether information should be stored,
ignored, updated, merged, superseded, reinforced, connected, or marked
as uncertain/conflicting.


4. JEV - SEMANTIC DECISION LAYER
--------------------------------

JEV acts as the semantic decision-making layer.

It reasons over candidate memories and existing memory state to determine
what should happen.

Possible decisions include:

    STORE
    IGNORE
    UPDATE
    MERGE
    SUPERSEDE
    REINFORCE
    CONFLICT
    CREATE_RELATION
    UNCERTAIN
    ESCALATE

A critical architectural rule is:

    JEV DECIDES
          |
          v
    MEMORY MANAGER MUTATES
          |
          v
    CANONICAL MEMORY

JEV should not directly mutate storage.

This separation keeps semantic reasoning independent from state mutation
and makes decisions auditable and testable.


5. MEMORY MANAGER
-----------------

The Memory Manager owns mutations to canonical memory.

It is responsible for safely applying semantic decisions such as:

    - Create
    - Update
    - Merge
    - Supersede
    - Reinforce
    - Decay
    - Archive
    - Split
    - Create relationship
    - Remove relationship

It also maintains:

    - Versions
    - Provenance
    - Lifecycle state
    - Stable identities
    - Consistency rules

This prevents different components from directly modifying the semantic
state of the system.


6. EVIDENCE / OBJECT STORE
--------------------------

A memory should remain connected to the experience from which it came.

The Object Store preserves durable evidence such as:

    - Original messages
    - Episodes
    - Raw source information
    - Evidence references
    - Snapshots

The lineage is therefore:

    MEMORY
       |
       v
    PROVENANCE
       |
       v
    EVIDENCE
       |
       v
    ORIGINAL EXPERIENCE

This makes memories traceable and allows the system to inspect original
evidence when necessary.


7. VECTOR MEMORY
----------------

Vector memory provides semantic retrieval.

A memory can be represented as an embedding so that semantically related
memories can be discovered efficiently.

The planned vector store is:

    QDRANT

However:

    VECTOR SIMILARITY != TRUTH

Vector similarity is used to generate candidates.

It does not automatically mean:

    - The memories are duplicates
    - The memories represent the same fact
    - The memory is currently valid
    - One memory should replace another
    - The information is semantically equivalent

Final semantic decisions are made using the broader memory architecture.


8. GRAPH MEMORY
---------------

Graph memory represents relationships that cannot always be captured
effectively through vector similarity.

The planned graph store is:

    NEO4J

Important graph objects include:

    - Memory
    - Entity
    - Episode
    - Project

Possible relationships include:

    Memory -> Entity
    Memory -> Memory
    Entity -> Entity
    Memory -> Episode

Example relationship types:

    RELATED_TO
    SUPPORTS
    CONTRADICTS
    SUPERSEDES
    DERIVED_FROM
    REINFORCES
    SAME_CONTEXT
    PART_OF
    USES
    DEPENDS_ON
    CREATED_BY

Graph relationships can also contain confidence, provenance,
inference type, validity, and source references.

Graph proximity is also not treated as semantic truth.


9. TEMPORAL MEMORY
------------------

Memory is not static.

Facts, preferences, project states, relationships, and decisions can
change over time.

Temporal information is therefore treated as a first-class part of the
memory system.

Important temporal concepts include:

    CREATED_AT
    LEARNED_AT
    EVENT_TIME
    VALID_FROM
    VALID_UNTIL

This allows the system to answer questions such as:

    - What is true now?
    - What was true previously?
    - When did something change?
    - What happened before an event?
    - What happened after an event?
    - Which state was valid during a particular period?

Historical information is preserved instead of blindly overwritten.


======================================================================
HYBRID RETRIEVAL
======================================================================

The retrieval system does not depend on a single retrieval mechanism.

A query can be processed using current memory, vector search, graph
search, and temporal reasoning.

                             QUERY
                               |
                               v
                        QUERY ANALYZER
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
        CURRENT MEMORY    VECTOR SEARCH    GRAPH SEARCH
              |                |                |
              +----------------+----------------+
                               |
                               v
                       CANDIDATE FUSION
                               |
                               v
                      TEMPORAL FILTERING
                               |
                               v
                          RE-RANKING
                               |
                               v
                       CONTEXT BUILDER
                               |
                               v
                           LLM AGENT

Retrieval can therefore combine:

    - Semantic similarity
    - Graph relevance
    - Temporal validity
    - Current state
    - Importance
    - Confidence
    - Reinforcement
    - Historical context

The goal is not to retrieve the maximum amount of information.

The goal is to retrieve the smallest useful set of memories required
to answer the current task correctly.


======================================================================
CONTEXT BUILDER
======================================================================

Retrieved memories still need to be converted into useful context before
being provided to the LLM.

The Context Builder is responsible for:

    - Removing unnecessary duplicates
    - Resolving current versus historical states
    - Grouping related memories
    - Preserving important provenance
    - Prioritizing relevant information
    - Respecting context limits

The final flow is:

    MEMORY SYSTEM
         |
         v
    RETRIEVED MEMORIES
         |
         v
    CONTEXT BUILDER
         |
         v
    USEFUL AGENT CONTEXT
         |
         v
    LLM


======================================================================
DREAM ENGINE
======================================================================

The Dream Engine is the asynchronous memory evolution component.

Unlike normal memory formation, which processes new experiences, the
Dream Engine examines memories that already exist.

Its purpose is to continuously improve the organization and quality of
long-term memory.

The general flow is:

    MEMORY EVENTS
         |
         v
    DREAM QUEUE
         |
         v
    CANDIDATE SELECTION
         |
         v
    NEIGHBORHOOD BUILDING
         |
         v
    CLUSTERING
         |
         v
    STRUCTURAL ANALYSIS
         |
         +-------------------+------------------+
         |                   |                  |
         v                   v                  v
      DUPLICATES          CONFLICTS       RELATIONSHIPS
         |                   |                  |
         +-------------------+------------------+
                             |
                             v
                     TEMPORAL REASONING
                             |
                             v
                        JEV DECISION
                             |
                             v
                      MEMORY MANAGER
                             |
                             v
                     CANONICAL MEMORY
                             |
                             v
                     INDEX SYNCHRONIZATION
                             |
                             v
                          SNAPSHOT

The Dream Engine can perform operations such as:

    - Duplicate detection
    - Memory consolidation
    - Conflict detection
    - Temporal state reasoning
    - Relationship discovery
    - Reinforcement
    - Decay
    - Archival
    - Memory splitting
    - Memory merging
    - Graph cleanup


======================================================================
MEMORY EVOLUTION
======================================================================

The system does not treat memory as immutable.

As new evidence arrives, existing memories can evolve through versioned
changes.

For example:

    MEMORY V1
       |
       | New Evidence
       v
    MEMORY V2
       |
       | State Change
       v
    MEMORY V3

Older states are not simply erased.

The system preserves lineage so that it can understand how knowledge
changed over time.

This is important for:

    - Changing project states
    - User preferences
    - Decisions
    - Relationships
    - Outdated information
    - Conflicting information
    - Historical reasoning


======================================================================
HIERARCHICAL MEMORY
======================================================================

The final system is designed to support different scopes and levels of
memory.

For example, an agent can maintain:

    GLOBAL USER MEMORY
            |
            v
       PROJECT MEMORY
            |
            v
     REPOSITORY MEMORY
            |
            v
      SESSION MEMORY
            |
            v
     WORKING MEMORY

Different memories have different relevance depending on the current
task.

A coding agent, for example, may prioritize:

    1. Current task / working context
    2. Current repository state
    3. Current project state
    4. Active architectural decisions
    5. Relevant historical information
    6. Broader long-term knowledge

This allows the system to preserve long-term information without
injecting everything into every LLM request.


======================================================================
AGENT INTEGRATION
======================================================================

The memory system is designed as an independent memory layer that an
LLM agent interacts with through a clean interface.

The agent should NOT directly manipulate:

    - Neo4j
    - Qdrant
    - Object Store
    - Internal canonical memory state

Instead:

                     LLM AGENT
                         |
                         v
                  MEMORY API / TOOLS
                         |
                         v
                  MEMORY ORCHESTRATOR
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
          FORMATION   RETRIEVAL    DREAMING
             |           |           |
             +-----------+-----------+
                         |
                         v
                  CANONICAL MEMORY

This keeps the memory system agent-agnostic and allows it to be used
with different agent frameworks and coding agents.


======================================================================
FIVE-PHASE DEVELOPMENT ARCHITECTURE
======================================================================


PHASE 1 - MEMORY KERNEL & FOUNDATION
------------------------------------

Question:

    "What is a memory?"

This phase establishes the semantic foundation of the system.

It defines:

    - Canonical Memory
    - Memory identity
    - Evidence
    - Provenance
    - Entities
    - Relationships
    - Temporal state
    - Lifecycle
    - Versioning
    - Confidence
    - Importance
    - Core interfaces
    - System invariants

This phase creates the foundation required by all later components.


PHASE 2 - MEMORY FORMATION
--------------------------

Question:

    "How does experience become memory?"

This phase builds the process that transforms raw experiences into
candidate and canonical memories.

It introduces:

    - Ingestion
    - Episodes
    - Messages
    - Working Memory
    - Candidate Memory
    - Entity Resolution
    - Temporal Extraction
    - Existing Memory Matching
    - JEV
    - Memory Manager
    - Formation decisions
    - Provenance
    - Domain events


PHASE 3 - STORAGE, GRAPH, VECTOR & RETRIEVAL
--------------------------------------------

Question:

    "Where is memory stored, connected, and retrieved?"

This phase introduces:

    - Object / Evidence Store
    - Qdrant Vector Store
    - Neo4j Graph Store
    - Vector indexing
    - Graph indexing
    - Hybrid retrieval
    - Temporal filtering
    - Candidate fusion
    - Re-ranking
    - Context construction
    - Retrieval modes
    - Rebuildable indexes

The goal is to complete the first meaningful vertical memory pipeline:

    EXPERIENCE
        |
        v
    FORMATION
        |
        v
    CANONICAL MEMORY
        |
        +------> OBJECT STORE
        |
        +------> QDRANT
        |
        +------> NEO4J
        |
        v
    RETRIEVAL
        |
        v
    CONTEXT
        |
        v
    LLM


PHASE 4 - DREAM ENGINE & MEMORY EVOLUTION
------------------------------------------

Question:

    "How does memory evolve as more experience arrives?"

This phase introduces asynchronous memory evolution.

It includes:

    - Dream Queue
    - Candidate Selection
    - Neighborhood Construction
    - Clustering
    - Duplicate Detection
    - Conflict Detection
    - Temporal Reasoning
    - Relationship Discovery
    - Consolidation
    - Reinforcement
    - Decay
    - Archival
    - Splitting
    - Version-aware mutations
    - Snapshots
    - Dream Reports


PHASE 5 - AGENT INTEGRATION, DEPLOYMENT & EVALUATION
----------------------------------------------------

Question:

    "How does an agent use the memory system, and does it actually work?"

This phase turns the memory architecture into a usable and measurable
agent memory system.

It includes:

    - Memory API
    - Agent-facing memory tools
    - Agent integration
    - Authentication and authorization
    - Scope-aware memory
    - Multi-tenancy
    - Docker deployment
    - Configuration management
    - Observability
    - Health monitoring
    - Recovery
    - Cost tracking
    - Benchmarking
    - Baselines
    - Ablation studies
    - Failure testing
    - Security testing
    - Scalability testing


======================================================================
COMPLETE DEVELOPMENT FLOW
======================================================================

                         PHASE 1
                  MEMORY KERNEL
                         |
                         v
                         PHASE 2
                  MEMORY FORMATION
                         |
                         v
                         PHASE 3
              STORAGE + GRAPH + VECTOR
                    + RETRIEVAL
                         |
                         v
                 VERTICAL MEMORY
                    PIPELINE
                         |
                         v
                         PHASE 4
                  DREAM ENGINE
                 + EVOLUTION
                         |
                         v
                         PHASE 5
                 AGENT INTEGRATION
                + DEPLOYMENT
                + EVALUATION
                         |
                         v
                  FINAL SYSTEM


======================================================================
CORE ARCHITECTURAL PRINCIPLES
======================================================================

1. CANONICAL MEMORY IS THE SOURCE OF TRUTH

   Semantic memory exists independently of its retrieval
   representations.


2. EVIDENCE MUST BE TRACEABLE

   Durable memories should maintain a connection to the evidence from
   which they originated.


3. VECTOR AND GRAPH ARE REPRESENTATIONS

   Qdrant and Neo4j provide retrieval and relationship capabilities,
   but do not independently define semantic truth.


4. SIMILARITY IS NOT TRUTH

   Vector similarity and graph proximity generate candidates.
   They do not automatically establish semantic equivalence.


5. JEV DECIDES, MEMORY MANAGER MUTATES

   Semantic reasoning and state mutation remain separate.


6. HISTORY MATTERS

   Memory changes should preserve previous states and lineage instead
   of blindly overwriting history.


7. TEMPORAL STATE IS FIRST-CLASS

   The system must distinguish current knowledge from historical
   knowledge.


8. DREAMING IS ASYNCHRONOUS

   Deep consolidation and memory evolution should not unnecessarily
   slow down normal agent interactions.


9. EXPENSIVE REASONING IS SELECTIVE

   Cheap deterministic and statistical filtering should happen before
   expensive reasoning or LLM escalation.


10. INDEXES SHOULD BE REBUILDABLE

    Vector and graph representations should be recoverable from
    canonical memory.


11. MEMORY IS SCOPE-AWARE

    Memory relevance depends on the current user, project, repository,
    session, and task.


12. VERSION-AWARE MUTATIONS

    Background processes such as dreaming must not overwrite newer
    changes made by the online system.


13. SAFE DEGRADATION

    Failure of a derived retrieval system should not corrupt canonical
    memory.


14. EXPERIMENTS MUST BE MEASURED

    Performance and quality claims will be based on actual experiments,
    not assumed or fabricated numbers.


======================================================================
RESEARCH & EVALUATION DIRECTION
======================================================================

This project is intended to be more than a memory database.

A major goal is to experimentally investigate whether combining
different memory mechanisms improves long-term agent memory.

The final evaluation will investigate questions such as:

    - Does graph memory improve multi-hop retrieval?
    - Does temporal reasoning improve current-versus-historical accuracy?
    - Does hierarchical memory reduce unnecessary context?
    - Does dreaming improve long-term memory quality?
    - Does reinforcement and decay improve active-memory quality?
    - Does selective LLM reasoning reduce memory-processing cost?
    - How does the system behave as memory grows?
    - What is the trade-off between quality, latency, tokens, and cost?

The evaluation will eventually include:

    - Baselines
    - Ablation studies
    - Controlled datasets
    - Ground-truth queries
    - Retrieval evaluation
    - Memory formation evaluation
    - Temporal evaluation
    - Relationship evaluation
    - Consolidation evaluation
    - Cost evaluation
    - Latency evaluation
    - Scalability evaluation

No benchmark numbers are claimed until the experiments are actually
implemented and executed.


======================================================================
TECHNOLOGY STACK
======================================================================

Initial planned stack:

    Python
        Core memory system and processing

    FastAPI
        Memory API and service interface

    Qdrant
        Vector storage and semantic retrieval

    Neo4j
        Graph storage and relationship retrieval

    Object / File Store
        Raw evidence, durable files and snapshots

    Docker
        Local infrastructure and reproducible environment

    LLM / Embedding Models
        Semantic reasoning, memory processing and vector representation

    Git / GitHub
        Version control and project development

The technology stack may evolve as implementation and experiments
provide evidence for better choices.


======================================================================
DEVELOPMENT ROADMAP
======================================================================

    PHASE 1
    Memory Kernel & Foundation
             |
             v
    PHASE 2
    Memory Formation & Decision Layer
             |
             v
    PHASE 3
    Storage + Graph + Vector + Retrieval
             |
             v
    COMPLETE VERTICAL MEMORY PIPELINE
             |
             v
    PHASE 4
    Dream Engine & Memory Evolution
             |
             v
    PHASE 5
    Agent Integration + Deployment
             |
             v
    EVALUATION
    + BENCHMARKS
    + BASELINES
    + ABLATIONS
             |
             v
    FINAL RESEARCH / DEMO SYSTEM


======================================================================
CURRENT STATUS
======================================================================

STATUS: IN DEVELOPMENT

Current phase:

    PHASE 1 - MEMORY KERNEL & FOUNDATION

The project is now beginning implementation of the foundational
architecture.

The immediate objective is to establish the core semantic contracts,
canonical memory model, evidence/provenance model, identity, lifecycle,
temporal information, versioning, and core interfaces.

Detailed implementation information will be added to this README as
each component is actually built, tested, and committed.


======================================================================
README DEVELOPMENT PHILOSOPHY
======================================================================

This README intentionally starts as a high-level architecture and
project overview.

As development progresses, the README will evolve together with the
actual implementation.

For each completed architecture component, the README will be expanded
with the relevant implementation details, including:

    - Architecture decisions
    - Module structure
    - Interfaces
    - Data models
    - APIs
    - Algorithms
    - Tests
    - Integration details
    - Screenshots
    - Experiments
    - Benchmarks
    - Failure cases
    - Observations
    - Results
    - Trade-offs
    - Lessons learned

Only implemented and experimentally verified results will be added.

This keeps the documentation aligned with the actual system instead of
documenting functionality before it exists.


======================================================================
FINAL VISION
======================================================================

The complete system can be viewed as:

                              EXPERIENCE
                                  |
                                  v
                         MEMORY FORMATION
                                  |
                                  v
                         CANONICAL MEMORY
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
                 EVIDENCE       VECTOR        GRAPH
                 / FILES        MEMORY        MEMORY
                    |             |             |
                    +-------------+-------------+
                                  |
                                  v
                          HYBRID RETRIEVAL
                                  |
                                  v
                         TEMPORAL REASONING
                                  |
                                  v
                           CONTEXT BUILDER
                                  |
                                  v
                              LLM AGENT
                                  |
                                  v
                           NEW EXPERIENCE
                                  |
                                  +----------------+
                                                   |
                                                   v
                                            DREAM ENGINE
                                                   |
                                                   v
                                            CONSOLIDATION
                                                   |
                                                   v
                                           MEMORY EVOLUTION
                                                   |
                                                   v
                                            BETTER MEMORY


The central idea of the project is:

    FORMATION CREATES MEMORY
    RETRIEVAL ACCESSES MEMORY
    DREAMING REORGANIZES MEMORY
    EVOLUTION IMPROVES MEMORY

Together, these components form a persistent, evidence-backed,
temporal, relational, hierarchical, and continuously evolving memory
system for LLM agents.
