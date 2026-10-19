/**
 * C test runner for numba_cfunc_compiler compiled functions.
 *
 * Python compiles the functions and passes function pointers here.
 * All test logic, assertions, and void* array management lives in C.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* cfunc signature:
 *   void (*)(void** outputs, int8_t* output_ticked,
 *            void** state, int8_t lifecycle_phase,
 *            void** inputs, int8_t* input_ticked, int8_t* input_valid)
 */
typedef void (*numba_cfunc_t)(void**, int8_t*, void**, int8_t, void**, int8_t*, int8_t*);

typedef enum {
  LIFECYCLE_EXECUTE = 0,
  LIFECYCLE_START   = 1,
  LIFECYCLE_STOP    = 2,
} LifecyclePhase;

/* ========================================================================
 * NumbaNode — clean C wrapper for calling compiled cfuncs
 * ======================================================================== */

#define MAX_SLOTS 8

typedef struct {
  numba_cfunc_t func;

  /* input storage (int64-sized slots, reinterpret for double/bool) */
  int64_t input_storage[MAX_SLOTS];
  void* inputs[MAX_SLOTS];
  int8_t ticked[MAX_SLOTS];
  int8_t valid[MAX_SLOTS];

  /* output storage */
  int64_t output_storage[MAX_SLOTS];
  void* outputs[MAX_SLOTS];
  int8_t out_ticked[MAX_SLOTS];

  /* state storage */
  int64_t state_storage[MAX_SLOTS];
  void* state[MAX_SLOTS];

  int n_inputs, n_outputs, n_state;
} NumbaNode;

void node_init(NumbaNode* n, void* fp, int ni, int no, int ns) {
  memset(n, 0, sizeof(NumbaNode));
  n->func      = (numba_cfunc_t)fp;
  n->n_inputs  = ni;
  n->n_outputs = no;
  n->n_state   = ns;
  for(int i = 0; i < ni; i++) {
    n->inputs[i] = &n->input_storage[i];
    n->ticked[i] = 1;
    n->valid[i]  = 1;
  }
  for(int i = 0; i < no; i++) {
    n->outputs[i] = &n->output_storage[i];
  }
  for(int i = 0; i < ns; i++) {
    n->state[i] = &n->state_storage[i];
  }
}

void node_set_int(NumbaNode* n, int idx, int64_t val) { n->input_storage[idx] = val; }
void node_set_double(NumbaNode* n, int idx, double val) { *(double*)&n->input_storage[idx] = val; }
void node_set_bool(NumbaNode* n, int idx, int8_t val) { *(int8_t*)&n->input_storage[idx] = val; }
void node_set_state(NumbaNode* n, int idx, int64_t val) { n->state_storage[idx] = val; }

void node_start(NumbaNode* n) {
  n->func(n->outputs, n->out_ticked, n->state, LIFECYCLE_START, n->inputs, n->ticked, n->valid);
}

void node_execute(NumbaNode* n) {
  for(int i = 0; i < n->n_outputs; i++)
    n->out_ticked[i] = 0;
  n->func(n->outputs, n->out_ticked, n->state, LIFECYCLE_EXECUTE, n->inputs, n->ticked, n->valid);
}

int64_t node_get_int(NumbaNode* n, int idx) { return n->output_storage[idx]; }
double node_get_double(NumbaNode* n, int idx) { return *(double*)&n->output_storage[idx]; }
int node_was_ticked(NumbaNode* n, int idx) { return n->out_ticked[idx] != 0; }

/* ========================================================================
 * Test framework
 * ======================================================================== */

static int _pass_count       = 0;
static int _fail_count       = 0;
static char _last_error[512] = "";

#define ASSERT_INT_EQ(actual, expected, msg)                                                                           \
  do {                                                                                                                 \
    if((actual) != (expected)) {                                                                                       \
      snprintf(                                                                                                        \
        _last_error, sizeof(_last_error), "%s: expected %lld, got %lld", msg, (long long)(expected),                   \
        (long long)(actual)                                                                                            \
      );                                                                                                               \
      _fail_count++;                                                                                                   \
      return;                                                                                                          \
    }                                                                                                                  \
    _pass_count++;                                                                                                     \
  } while(0)

#define ASSERT_DOUBLE_EQ(actual, expected, msg)                                                                        \
  do {                                                                                                                 \
    if(fabs((actual) - (expected)) > 1e-9) {                                                                           \
      snprintf(                                                                                                        \
        _last_error, sizeof(_last_error), "%s: expected %.6f, got %.6f", msg, (double)(expected), (double)(actual)     \
      );                                                                                                               \
      _fail_count++;                                                                                                   \
      return;                                                                                                          \
    }                                                                                                                  \
    _pass_count++;                                                                                                     \
  } while(0)

#define ASSERT_TRUE(cond, msg)                                                                                         \
  do {                                                                                                                 \
    if(!(cond)) {                                                                                                      \
      snprintf(_last_error, sizeof(_last_error), "%s: expected true", msg);                                            \
      _fail_count++;                                                                                                   \
      return;                                                                                                          \
    }                                                                                                                  \
    _pass_count++;                                                                                                     \
  } while(0)

#define ASSERT_FALSE(cond, msg)                                                                                        \
  do {                                                                                                                 \
    if(cond) {                                                                                                         \
      snprintf(_last_error, sizeof(_last_error), "%s: expected false", msg);                                           \
      _fail_count++;                                                                                                   \
      return;                                                                                                          \
    }                                                                                                                  \
    _pass_count++;                                                                                                     \
  } while(0)

/* ========================================================================
 * Tests — each receives a function pointer compiled by Python
 * ======================================================================== */

static void test_add_ints(void* fp) {
  NumbaNode n;
  node_init(&n, fp, 2, 1, 0);

  node_set_int(&n, 0, 10);
  node_set_int(&n, 1, 20);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), 30, "10 + 20");

  node_set_int(&n, 0, -5);
  node_set_int(&n, 1, 5);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), 0, "-5 + 5");

  node_set_int(&n, 0, 1000000);
  node_set_int(&n, 1, 2000000);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), 3000000, "1M + 2M");
}

static void test_add_floats(void* fp) {
  NumbaNode n;
  node_init(&n, fp, 2, 1, 0);

  node_set_double(&n, 0, 1.5);
  node_set_double(&n, 1, 2.5);
  node_execute(&n);
  ASSERT_DOUBLE_EQ(node_get_double(&n, 0), 4.0, "1.5 + 2.5");
}

static void test_multiply_constant(void* fp) {
  NumbaNode n;
  node_init(&n, fp, 1, 1, 0);

  node_set_int(&n, 0, 7);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), 21, "7 * 3");

  node_set_int(&n, 0, -4);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), -12, "-4 * 3");
}

static void test_conditional(void* fp) {
  NumbaNode n;
  node_init(&n, fp, 1, 1, 0);

  node_set_int(&n, 0, 5);
  node_execute(&n);
  ASSERT_FALSE(node_was_ticked(&n, 0), "5 < 10 should not tick");

  node_set_int(&n, 0, 15);
  node_execute(&n);
  ASSERT_TRUE(node_was_ticked(&n, 0), "15 > 10 should tick");
  ASSERT_INT_EQ(node_get_int(&n, 0), 15, "output = 15");
}

static void test_bool_signal(void* fp) {
  NumbaNode n;
  node_init(&n, fp, 2, 1, 0);

  node_set_int(&n, 0, 42);
  node_set_bool(&n, 1, 1);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), -42, "negate(42, true)");

  node_set_bool(&n, 1, 0);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), 42, "negate(42, false)");
}

static void test_stateful_accumulator(void* fp) {
  NumbaNode n;
  node_init(&n, fp, 1, 1, 1);
  node_set_state(&n, 0, 0);
  node_start(&n);

  node_set_int(&n, 0, 10);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), 10, "acc after 10");

  node_set_int(&n, 0, 5);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), 15, "acc after 10+5");

  node_set_int(&n, 0, 25);
  node_execute(&n);
  ASSERT_INT_EQ(node_get_int(&n, 0), 40, "acc after 10+5+25");
}

static void test_ema_benchmark(void* fp) {
  NumbaNode n;
  node_init(&n, fp, 1, 1, 1);
  node_set_state(&n, 0, 0);
  node_start(&n);

  int N = 100000;
  for(int i = 0; i < N; i++) {
    node_set_double(&n, 0, (double)i);
    node_execute(&n);
  }

  /* verify against reference */
  double ema = 0.0;
  for(int i = 0; i < N; i++)
    ema = 0.1 * (double)i + 0.9 * ema;

  ASSERT_DOUBLE_EQ(node_get_double(&n, 0), ema, "EMA 100k ticks");
}

/* ========================================================================
 * Entry point — called from Python with all function pointers
 * ======================================================================== */

typedef struct {
  void (*test_fn)(void* fp);
  const char* name;
} TestEntry;

int run_tests(
  void* fp_add_ints, void* fp_add_floats, void* fp_multiply, void* fp_conditional, void* fp_bool, void* fp_accumulator,
  void* fp_ema
) {
  TestEntry tests[] = {
    {test_add_ints,             "add_ints"            },
    {test_add_floats,           "add_floats"          },
    {test_multiply_constant,    "multiply_constant"   },
    {test_conditional,          "conditional"         },
    {test_bool_signal,          "bool_signal"         },
    {test_stateful_accumulator, "stateful_accumulator"},
    {test_ema_benchmark,        "ema_benchmark"       },
  };
  void* fps[] = {
    fp_add_ints, fp_add_floats, fp_multiply, fp_conditional, fp_bool, fp_accumulator, fp_ema,
  };

  _pass_count    = 0;
  _fail_count    = 0;
  _last_error[0] = '\0';

  int n_tests = sizeof(tests) / sizeof(tests[0]);
  for(int i = 0; i < n_tests; i++) {
    int prev_fail = _fail_count;
    tests[i].test_fn(fps[i]);
    if(_fail_count > prev_fail)
      printf("  FAIL: %s — %s\n", tests[i].name, _last_error);
    else
      printf("  PASS: %s\n", tests[i].name);
  }

  printf("\n  %d passed, %d failed\n", _pass_count, _fail_count);
  return _fail_count;
}

/* Also expose individual pieces for Python to read */
int get_pass_count(void) { return _pass_count; }
int get_fail_count(void) { return _fail_count; }
const char* get_last_error(void) { return _last_error; }
