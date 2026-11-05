"""Demonstration of the tokenizer system.

This script shows how to:
1. Create tokenizer specs
2. Instantiate tokenizers from specs
3. Use the registry
4. Work with contributed tokenizers
"""

import torch
from ntt.core.model.tokenizer.contrib.identity_tokenizer import (
    IdentityTokenizerSpec,
)

from ntt.core.model.tokenizer import (
    create_tokenizer,
    get_tokenizer_class,
    list_tokenizers,
)


def main():
    print("=" * 60)
    print("NTT Tokenizer System Demo")
    print("=" * 60)

    # 1. List all registered tokenizers
    print("\n1. Available tokenizers:")
    for name in list_tokenizers():
        print(f"   - {name}")

    # 2. Create a spec
    print("\n2. Creating a tokenizer spec:")
    spec = IdentityTokenizerSpec(
        window_length_seconds=1.0,
        bin_size_seconds=0.01,
        n_neurons=100,
        vocab_size=1024,
    )
    print(f"   Spec: {spec}")
    print(f"   Vocab size: {spec.get_vocab_size()}")

    # 3. Instantiate tokenizer from spec
    print("\n3. Creating tokenizer from spec:")
    tokenizer = create_tokenizer(spec)
    print(f"   Tokenizer: {tokenizer.__class__.__name__}")
    print(f"   Tokenizer.spec: {tokenizer.spec}")

    # 4. Use the tokenizer
    print("\n4. Using the tokenizer:")
    # Create dummy input (batch_size=2, time=100, neurons=50)
    x = torch.randn(2, 100, 50)
    print(f"   Input shape: {x.shape}")

    # Forward pass
    output = tokenizer(x)
    print(f"   Output keys: {list(output.keys())}")
    print(f"   Tokens shape: {output['tokens'].shape}")
    print(f"   Reconstruction shape: {output['reconstruction'].shape}")

    # Individual encode/decode
    tokens = tokenizer.encode(x)
    reconstruction = tokenizer.decode(tokens)
    print(f"   Encode -> tokens: {tokens.shape}")
    print(f"   Decode -> reconstruction: {reconstruction.shape}")

    # 5. Access registry directly
    print("\n5. Using the registry:")
    spec_class, tokenizer_class = get_tokenizer_class("identity")
    print("   Registry for 'identity':")
    print(f"     Spec class: {spec_class.__name__}")
    print(f"     Tokenizer class: {tokenizer_class.__name__}")

    # 6. Access datamodule-relevant spec attributes
    print("\n6. Spec attributes for datamodule configuration:")
    print(f"   window_length_seconds: {spec.window_length_seconds}")
    print(f"   bin_size_seconds: {spec.bin_size_seconds}")
    print(f"   n_neurons: {spec.n_neurons}")
    print(f"   shuffling_strategy: {spec.shuffling_strategy}")
    print(f"   sampling_strategy: {spec.sampling_strategy}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
