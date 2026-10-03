{
  description = "exllamav3 (ROCm/RDNA3 port) + TabbyAPI on NixOS, tuned for a 24 GB RX 7900 XTX (gfx1100)";

  inputs = {
    # Pinned to a revision whose python313 torchWithRocm (2.11, ROCm 7.2.3) is in cache.nixos.org
    nixpkgs.url = "github:NixOS/nixpkgs/5e2305d577ca00acbba631b05cb1094d172b29f3";
    tabbyapi = {
      url = "github:theroyallab/tabbyAPI/f07131cd8fe34e449fe87cdd3a066b52b96d3cac";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, tabbyapi }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      inherit (pkgs) lib;
      rp = pkgs.rocmPackages;
      python = pkgs.python313;
      pp = python.pkgs;
      torch = pp.torchWithRocm;
      gpuTarget = "gfx1100";

      # /opt/rocm-like tree for torch.utils.cpp_extension: hipcc, HIP + library headers, device bitcode
      rocmHome = pkgs.symlinkJoin {
        name = "rocm-home";
        paths = lib.concatMap
          (p: builtins.filter (o: builtins.elem o.outputName [ "out" "dev" ]) (p.all or [ p ]))
          (with rp; [
            clr rocm-core hipblas hipblas-common hipblaslt hipsparse hipsolver hiprand rocrand
            rocblas rocprim hipcub rocthrust rocsolver rocsparse roctracer rocm-device-libs
          ]);
      };
      rocmEnv = {
        ROCM_HOME = "${rocmHome}";
        ROCM_PATH = "${rocmHome}";
        HIP_PATH = "${rocmHome}";
        HIP_DEVICE_LIB_PATH = "${rocmHome}/amdgcn/bitcode";
        PYTORCH_ROCM_ARCH = gpuTarget;
        # nixpkgs torch (unlike the pip wheel) does not vendor pybind11's headers
        CPLUS_INCLUDE_PATH = "${pp.pybind11}/include";
      };
      # Host C++ must go through ROCm clang (g++ cannot parse the HIP bf16 headers); stdenv's
      # setup overwrites CC/CXX, so they are exported in the build phase / shell hook instead
      useRocmClang = ''
        export CC=${rp.llvm.clang}/bin/clang CXX=${rp.llvm.clang}/bin/clang++
      '';

      exlDeps = p: with p; [
        torchWithRocm tokenizers numpy rich typing-extensions safetensors ninja pillow pyyaml
        marisa-trie pydantic llguidance
      ];

      exllamav3 = pp.buildPythonPackage {
        pname = "exllamav3";
        version = "1.5.1+rocm.${gpuTarget}";
        pyproject = true;
        src = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [ ./exllamav3 ./setup.py ./pyproject.toml ./README.md ./LICENSE ];
        };
        build-system = [ pp.setuptools pp.wheel pp.ninja torch ];
        nativeBuildInputs = [ pkgs.ninja ];
        dependencies = exlDeps pp;
        env = rocmEnv;
        preBuild = useRocmClang + ''
          export MAX_JOBS=$NIX_BUILD_CORES
        '';
        # Importing the extension needs a GPU (absent in the build sandbox); check the .so instead
        doCheck = false;
        postInstall = ''
          ls $out/${python.sitePackages}/exllamav3_ext*.so
        '';
      };

      tabbyapiSrc = pkgs.applyPatches {
        name = "tabbyapi-rdna3";
        src = tabbyapi;
        patches = [ ./rocm/tabbyapi/0001-exllamav3-allow-rdna3.patch ];
      };

      serverPython = python.withPackages (p: [ exllamav3 ] ++ (with p; [
        fastapi pydantic ruamel-yaml rich uvicorn jinja2 loguru sse-starlette packaging tokenizers
        numpy aiofiles aiohttp async-lru huggingface-hub psutil httptools pillow requests uvloop
        setuptools formatron kbnf
      ]));

      # Serving profiles, measured on a 7900 XTX with the turboderp 4.0bpw quant (SC_4.00bpw_H5)
      # and two concurrent coding jobs (see rocm_tests/bench_slots.py):
      #   fast: DFlash2 draft, 128K shared pool;  ~62-104 tok/s single, 120K session + 8K side job fit
      #   long: MTP draft, 208K shared pool;      ~45-69 tok/s single, 180K session + 8K side job fit
      # The KV pool is shared and paged: one request may use (almost) all of it, two share it.
      mkProfile = name: m: d: (pkgs.formats.yaml { }).generate "tabby-${name}.yml" {
        network = { host = "127.0.0.1"; port = 8096; disable_auth = true; api_servers = [ "OAI" ]; };
        logging = { log_prompt = false; log_generation_params = false; log_requests = false; };
        model = {
          model_dir = "models";
          backend = "exllamav3";
          cache_mode = "Q8";
          chunk_size = 2048;
          max_batch_size = 2;
          gpu_split_auto = true;
          autosplit_reserve = [ 512 ];
          vision = false;
          reasoning = true;
          reasoning_start_token = "<think>";
          reasoning_end_token = "</think>";
          tool_format = "qwen3_coder";
        } // m;
        draft_model = { draft_cache_mode = "Q4"; } // d;
        memory.sysmem_recurrent_cache = 4096;
      };
      profiles = {
        fast = mkProfile "fast"
          { cache_size = 131072; max_seq_len = 131072; }
          { draft_mode = "model"; draft_model_dir = "models"; };
        long = mkProfile "long"
          { cache_size = 212992; max_seq_len = 212992; }
          { draft_mode = "mtp"; };
      };

      exllama = pkgs.writeShellApplication {
        name = "exllama";
        runtimeInputs = [ pkgs.coreutils ];
        text = ''
          usage() {
            cat <<EOF
          usage: exllama [--profile fast|long] [--models DIR] [--model NAME] [--draft NAME]
                         [--state DIR] [TABBYAPI-ARGS...]

            --profile  fast (default): DFlash2 drafting, 2 slots sharing a 128K-token KV pool
                       long:           MTP drafting,     2 slots sharing a 208K-token KV pool
            --models   directory holding the model folders   (default: \$EXLLAMA_MODELS or ./models)
            --model    main model folder name                 (default: Qwen3.8-27B-exl3-SC4.0)
            --draft    DFlash2 draft folder name (fast only)  (default: Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
            --state    writable TabbyAPI state dir            (default: \''${XDG_STATE_HOME:-~/.local/state}/exllama)

          Remaining arguments go to TabbyAPI and override the profile, e.g. --port 5000,
          --host 0.0.0.0 --disable-auth false, --cache-size 98304. Listens on 127.0.0.1:8096.
          EOF
          }
          profile=fast
          models=''${EXLLAMA_MODELS:-$PWD/models}
          model=Qwen3.8-27B-exl3-SC4.0
          draft=Qwen3.8-27B-DFlash2-EXL3-5.0bpw
          state=''${XDG_STATE_HOME:-$HOME/.local/state}/exllama
          passthru=()
          while [ $# -gt 0 ]; do
            case "$1" in
              --profile) profile=$2; shift 2 ;;
              --models) models=$2; shift 2 ;;
              --model) model=$2; shift 2 ;;
              --draft) draft=$2; shift 2 ;;
              --state) state=$2; shift 2 ;;
              -h|--help) usage; exit 0 ;;
              *) passthru+=("$1"); shift ;;
            esac
          done
          case "$profile" in
            fast) config=${profiles.fast}; passthru=(--draft-model-name "$draft" "''${passthru[@]}") ;;
            long) config=${profiles.long} ;;
            *) echo "unknown profile: $profile" >&2; usage >&2; exit 2 ;;
          esac
          models=$(realpath "$models")
          [ -d "$models/$model" ] || { echo "model not found: $models/$model" >&2; exit 1; }

          # TabbyAPI resolves config.yml, templates/, sampler_overrides/, logs/ and api_tokens.yml
          # relative to the working directory: run it from a writable state dir
          mkdir -p "$state"
          ln -sfn ${tabbyapiSrc}/templates "$state/templates"
          ln -sfn ${tabbyapiSrc}/sampler_overrides "$state/sampler_overrides"
          ln -sfn "$config" "$state/config.yml"
          ln -sfn "$models" "$state/models"
          cd "$state"

          export EXL3_NOGRAPH=''${EXL3_NOGRAPH-mlp,gdn}  # eager MLP/DeltaNet decode (port's tuned default)
          export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-8}
          # Compiled Triton kernels persist across restarts; also avoids torch's getpass() lookup
          export TRITON_CACHE_DIR=''${TRITON_CACHE_DIR:-$state/cache/triton}
          export TORCHINDUCTOR_CACHE_DIR=''${TORCHINDUCTOR_CACHE_DIR:-$state/cache/inductor}
          exec ${serverPython}/bin/python ${tabbyapiSrc}/main.py --model-name "$model" "''${passthru[@]}"
        '';
      };
    in
    {
      packages.${system} = {
        inherit exllamav3 exllama;
        tabbyapi = tabbyapiSrc;
        default = exllama;
      };

      apps.${system} = {
        exllama = { type = "app"; program = lib.getExe exllama; };
        default = self.apps.${system}.exllama;
      };

      # Development: same toolchain and Python deps; build the extension in place with
      #   python setup.py build_ext --inplace   (then run rocm_tests/* from the repo)
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          (python.withPackages (p: exlDeps p ++ (with p; [ setuptools wheel huggingface-hub hf-xet ])))
          pkgs.ninja pkgs.git rp.rocminfo rp.rocm-smi
        ];
        env = rocmEnv // { OMP_NUM_THREADS = "8"; };
        shellHook = useRocmClang + ''
          export PYTHONPATH=$PWD''${PYTHONPATH:+:$PYTHONPATH}
        '';
      };
    };
}
