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

      # The HIP extension is its own derivation with only its sources as input: Python, README or
      # test edits do not recompile it (the compile takes ~25 min on 8 threads)
      exllamav3Ext = pkgs.stdenv.mkDerivation {
        pname = "exllamav3-ext";
        version = "1.5.1+rocm.${gpuTarget}";
        src = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [ ./setup.py ./exllamav3/exllamav3_ext ];
        };
        nativeBuildInputs = [ pkgs.ninja (python.withPackages (p: [ p.setuptools p.ninja torch ])) ];
        env = rocmEnv;
        buildPhase = useRocmClang + ''
          export MAX_JOBS=$NIX_BUILD_CORES
          python setup.py build_ext --inplace
        '';
        installPhase = ''
          install -Dm755 -t $out/${python.sitePackages} exllamav3_ext*.so
        '';
      };

      # Pure-Python package (EXLLAMA_NOCOMPILE), rebuilt in seconds; the prebuilt extension is linked
      # in as the top-level exllamav3_ext module that exllamav3/ext.py imports
      exllamav3 = pp.buildPythonPackage {
        pname = "exllamav3";
        version = "1.5.1+rocm.${gpuTarget}";
        pyproject = true;
        src = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [ ./exllamav3 ./setup.py ./pyproject.toml ./README.md ./LICENSE ];
        };
        build-system = [ pp.setuptools pp.wheel ];
        dependencies = exlDeps pp;
        env.EXLLAMA_NOCOMPILE = "1";
        # Importing the extension needs a GPU (absent in the build sandbox)
        doCheck = false;
        postInstall = ''
          ln -s ${exllamav3Ext}/${python.sitePackages}/exllamav3_ext*.so $out/${python.sitePackages}/
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

      # Serving profiles: one TabbyAPI config per main model and drafting mode. The KV pool is shared
      # and paged: one request may use (almost) all of it, two share it. Pool sizes (tokens, Q8 KV,
      # 2 slots, vision in system RAM, DFlash2 KV Q4) are the largest that survive prefill of a near-full
      # pool on a 7900 XTX, from rocm_tests/bench_slots.py runs and measured per-model free VRAM
      # (README "Model comparison"):
      #   fast:  DFlash2 draft  (fastest decode; draft KV + 1.6 GB of draft weights cost context)
      #   long:  MTP draft      (built-in head, ~1/4 slower than fast on code, more context)
      #   plain: no drafting    (~40 tok/s; most context, capped at 256K per request)
      mainModels = {
        sc4     = { dir = "Qwen3.8-27B-exl3-SC4.0";                      fast = 124; long = 192; plain = 244; };
        mia35   = { dir = "Qwen3.8-27B-EXL3-3.5bpw";                     fast = 160; long = 232; plain = 276; };
        swift35 = { dir = "Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6"; fast = 172; long = 240; plain = 292; };
        swift40 = { dir = "Swift-1.5-Qwen3.8-27B-exl3-SC_4.00bpw_H5_V6"; fast = 120; long = 192; plain = 236; };
      };
      draftModes = {
        fast = { draft_mode = "model"; draft_model_dir = "models"; };
        long = { draft_mode = "mtp"; };
        plain = { };
      };
      maxSeqLen = 262144;  # max_position_embeddings of Qwen3.8

      # Sampler fallbacks for requests that omit them: Qwen's recommended thinking-mode settings
      # (model card; reasoning is on in every profile). Not forced, so clients may override, e.g. with
      # the non-thinking set: temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5.
      samplerPreset = "qwen-thinking";
      samplerOverrides = pkgs.runCommand "tabby-sampler-overrides" { } ''
        mkdir $out
        cp ${tabbyapiSrc}/sampler_overrides/*.yml $out/
        cp ${(pkgs.formats.yaml { }).generate "${samplerPreset}.yml" (lib.mapAttrs (_: v: { override = v; force = false; }) {
          temperature = 1.0; top_p = 0.95; top_k = 20; min_p = 0.0;
          presence_penalty = 0.0; repetition_penalty = 1.0;
        })} $out/${samplerPreset}.yml
      '';
      mkProfile = key: mode: let pool = mainModels.${key}.${mode} * 1024; in
        (pkgs.formats.yaml { }).generate "tabby-${key}-${mode}.yml" ({
          network = { host = "127.0.0.1"; port = 8096; disable_auth = true; api_servers = [ "OAI" ]; };
          logging = { log_prompt = false; log_generation_params = false; log_requests = false; };
          model = {
            model_dir = "models";
            backend = "exllamav3";
            cache_mode = "Q8";
            cache_size = pool;
            max_seq_len = lib.min pool maxSeqLen;
            chunk_size = 2048;
            max_batch_size = 2;
            gpu_split_auto = true;
            autosplit_reserve = [ 512 ];
            vision = true;
            vision_offload = true;  # encoder weights in pinned system RAM: no VRAM left beside the KV pool
            reasoning = true;
            reasoning_start_token = "<think>";
            reasoning_end_token = "</think>";
            tool_format = "qwen3_coder";
          };
          memory.sysmem_recurrent_cache = 4096;
          sampling.override_preset = samplerPreset;
        } // lib.optionalAttrs (draftModes.${mode} != { }) {
          draft_model = { draft_cache_mode = "Q4"; } // draftModes.${mode};
        });
      # Shell case arms "<model>:<mode>) dir=...; config=...; pool=... ;;" for every combination
      profileCases = lib.concatStrings (lib.flatten (lib.mapAttrsToList (key: m:
        map (mode: ''
          ${key}:${mode}) dir=${m.dir}; config=${mkProfile key mode}; pool=${toString m.${mode}}K ;;
        '') (lib.attrNames draftModes)) mainModels));
      pad = n: s: s + lib.fixedWidthString (n - lib.stringLength s) " " "";
      modelHelp = lib.concatStrings (lib.mapAttrsToList (key: m: ''
        ${pad 9 key}${pad 45 m.dir}fast ${toString m.fast}K, long ${toString m.long}K, plain ${toString m.plain}K
      '') mainModels);

      exllama = pkgs.writeShellApplication {
        name = "exllama";
        runtimeInputs = [ pkgs.coreutils ];
        text = ''
          usage() {
            cat <<EOF
          usage: exllama [--profile fast|long|plain] [--model KEY] [--models DIR] [--draft NAME]
                         [--state DIR] [TABBYAPI-ARGS...]

            --profile  fast (default): DFlash2 drafting
                       long:           MTP drafting
                       plain:          no drafting
                       2 slots share one KV pool sized per model and profile (table below)
            --model    main model key (default: sc4):
          ${modelHelp}
            --models   directory holding the model folders   (default: \$EXLLAMA_MODELS or ./models)
            --draft    DFlash2 draft folder name (fast only)  (default: Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
            --state    writable TabbyAPI state dir            (default: \''${XDG_STATE_HOME:-~/.local/state}/exllama)

          Remaining arguments go to TabbyAPI and override the profile, e.g. --port 5000,
          --host 0.0.0.0 --disable-auth false, --cache-size 98304. Listens on 127.0.0.1:8096.
          EOF
          }
          profile=fast
          models=''${EXLLAMA_MODELS:-$PWD/models}
          model=sc4
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
          case "$model:$profile" in
          ${profileCases}
            *) echo "unknown model/profile: $model/$profile" >&2; usage >&2; exit 2 ;;
          esac
          [ "$profile" = fast ] && passthru=(--draft-model-name "$draft" "''${passthru[@]}")
          models=$(realpath "$models")
          [ -d "$models/$dir" ] || { echo "model not found: $models/$dir" >&2; exit 1; }
          echo "exllama: $dir, profile $profile, $pool KV pool" >&2

          # TabbyAPI resolves config.yml, templates/, sampler_overrides/, logs/ and api_tokens.yml
          # relative to the working directory: run it from a writable state dir
          mkdir -p "$state"
          ln -sfn ${tabbyapiSrc}/templates "$state/templates"
          ln -sfn ${samplerOverrides} "$state/sampler_overrides"
          ln -sfn "$config" "$state/config.yml"
          ln -sfn "$models" "$state/models"
          cd "$state"

          export EXL3_NOGRAPH=''${EXL3_NOGRAPH-mlp,gdn}  # eager MLP/DeltaNet decode (port's tuned default)
          export OMP_NUM_THREADS=''${OMP_NUM_THREADS:-8}
          # Compiled Triton kernels persist across restarts; also avoids torch's getpass() lookup
          export TRITON_CACHE_DIR=''${TRITON_CACHE_DIR:-$state/cache/triton}
          export TORCHINDUCTOR_CACHE_DIR=''${TORCHINDUCTOR_CACHE_DIR:-$state/cache/inductor}
          exec ${serverPython}/bin/python ${tabbyapiSrc}/main.py --model-name "$dir" "''${passthru[@]}"
        '';
      };
    in
    {
      packages.${system} = {
        inherit exllamav3 exllama;
        exllamav3-ext = exllamav3Ext;
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
