# Copyright (C) Microsoft Corporation. All rights reserved.
# This file is distributed under the University of Illinois Open Source License. See LICENSE.TXT for details.
# TableGen HLSL options.
set(LLVM_TARGET_DEFINITIONS HLSLOptions.td)
tablegen(LLVM HLSLOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(TablegenHLSLOptions)
