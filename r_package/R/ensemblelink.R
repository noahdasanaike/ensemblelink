#' @importFrom reticulate py_module_available import source_python py_install use_python use_condaenv
NULL

# Package environment to store Python module reference
.pkg_env <- new.env(parent = emptyenv())

#' Configure Python Environment
#'
#' Set up the Python environment for ensemblelink. Call this before using
#' ensemble_link() if you need to specify a particular Python installation.
#' If called with no arguments, will use the "r-ensemblelink" virtualenv
#' created by install_ensemblelink().
#'
#' @param python Path to Python executable, or NULL
#' @param condaenv Name of conda environment to use, or NULL
#' @param virtualenv Name or path of virtualenv. Default: "r-ensemblelink"
#'
#' @return Invisibly returns TRUE on success
#' @export
#'
#' @examples
#' \dontrun{
#' configure_python()  # Use default r-ensemblelink virtualenv
#' configure_python(condaenv = "myenv")  # Use conda environment
#' }
configure_python <- function(python = NULL, condaenv = NULL, virtualenv = "r-ensemblelink") {
  if (!is.null(condaenv)) {
    reticulate::use_condaenv(condaenv, required = TRUE)
  } else if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
  } else if (!is.null(virtualenv)) {
    reticulate::use_virtualenv(virtualenv, required = TRUE)
  }
  .pkg_env$configured <- TRUE
  invisible(TRUE)
}

#' Install Python Dependencies
#'
#' Install required Python packages for ensemblelink.
#'
#' @param method Installation method: "auto", "conda", or "virtualenv"
#' @param conda Path to conda executable (if method = "conda")
#' @param envname Name of environment to create/use
#' @param gpu Logical; install GPU-enabled packages (faiss-gpu, CUDA torch)
#'
#' @return Invisibly returns TRUE on success
#' @export
#'
#' @examples
#' \dontrun{
#' install_ensemblelink()
#' install_ensemblelink(method = "conda", envname = "ensemblelink", gpu = TRUE)
#' }
install_ensemblelink <- function(method = "auto", conda = "auto", envname = "r-ensemblelink", gpu = FALSE) {
  packages <- c(
    "numpy",
    "pandas",
    "torch",
    "sentence-transformers",
    "scikit-learn",
    "tqdm",
    "einops"
  )

  if (gpu) {
    packages <- c(packages, "faiss-gpu")
  } else {
    packages <- c(packages, "faiss-cpu")
  }

  reticulate::py_install(
    packages = packages,
    envname = envname,
    method = method,
    conda = conda,
    pip = TRUE
  )

  message("Python dependencies installed successfully.")
  message("You may need to restart R and call configure_python() to use the new environment.")

  invisible(TRUE)
}


#' Initialize Python Backend
#'
#' Internal function to load the Python module
#' @keywords internal
.init_python <- function() {
  if (is.null(.pkg_env$matcher)) {
    # Configure Python environment if not already done
    if (is.null(.pkg_env$configured) || !.pkg_env$configured) {
      configure_python()
    }

    # Get path to bundled Python code
    python_path <- system.file("python", "matcher.py", package = "ensemblelink")

    if (python_path == "") {
      stop("Could not find Python matcher module. Package may not be installed correctly.")
    }

    # Source the Python code
    reticulate::source_python(python_path)

    # Store reference to the matcher class
    .pkg_env$matcher <- EnsembleMatcher
    .pkg_env$initialized <- TRUE
  }
}


#' Zero-Shot Record Linkage
#'
#' Link records from a query dataset to a reference corpus using ensemble
#' retrieval (dense + sparse) and cross-encoder reranking. Requires no
#' labeled training data.
#'
#' @param queries Character vector of query strings to match
#' @param corpus Character vector of reference strings to match against
#' @param embedding_model Name of sentence-transformers model for embeddings.
#'   Default: "Qwen/Qwen3-Embedding-0.6B"
#' @param reranker_model Name of cross-encoder model for reranking.
#'   Default: "jinaai/jina-reranker-v3"
#' @param top_k Number of candidates to retrieve before reranking. Default: 30
#' @param return_scores Logical; if TRUE, return match scores. Default: FALSE
#' @param show_progress Logical; show progress bar. Default: TRUE
#' @param device Device for inference: "cuda", "cpu", or "auto". Default: "auto"
#'
#' @return A data frame with columns:
#'   \item{query}{Original query string}
#'   \item{match}{Best matching reference string}
#'   \item{score}{Match score (if return_scores = TRUE)}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Basic usage
#' queries <- c("New York City", "Los Angelas", "Chcago")
#' corpus <- c("New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX")
#' results <- ensemble_link(queries, corpus)
#'
#' # With scores
#' results <- ensemble_link(queries, corpus, return_scores = TRUE)
#'
#' # Custom models
#' results <- ensemble_link(
#'   queries, corpus,
#'   embedding_model = "BAAI/bge-small-en-v1.5",
#'   reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#' )
#' }
ensemble_link <- function(
    queries,
    corpus,
    embedding_model = "Qwen/Qwen3-Embedding-0.6B",
    reranker_model = "jinaai/jina-reranker-v3",
    top_k = 30L,
    return_scores = FALSE,
    show_progress = TRUE,
    device = "auto"
) {
  # Validate inputs
  if (!is.character(queries) || length(queries) == 0) {
    stop("'queries' must be a non-empty character vector")
  }
  if (!is.character(corpus) || length(corpus) == 0) {
    stop("'corpus' must be a non-empty character vector")
  }

  # Initialize Python backend
  .init_python()

  # Create matcher instance
  matcher <- .pkg_env$matcher(
    embedding_model = embedding_model,
    reranker_model = reranker_model,
    top_k = as.integer(top_k),
    device = device
  )

  # Index corpus
  if (show_progress) message("Indexing corpus (", length(corpus), " records)...")
  matcher$index(corpus, show_progress = show_progress)

  # Match queries
  if (show_progress) message("Matching ", length(queries), " queries...")
  results <- matcher$match(queries, return_scores = return_scores, show_progress = show_progress)

  # Convert to data frame
  if (return_scores) {
    df <- data.frame(
      query = queries,
      match = results[[1]],
      score = results[[2]],
      stringsAsFactors = FALSE
    )
  } else {
    df <- data.frame(
      query = queries,
      match = results,
      stringsAsFactors = FALSE
    )
  }

  return(df)
}
