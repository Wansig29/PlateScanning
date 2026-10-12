r"""Sign a release installer so installed scanners will accept it as an in-app update.

Run this on your own computer, never in GitHub Actions: the point is that the
private key is not on GitHub, so someone who gets into the GitHub account can
publish a release but can't make the scanners install it.

One time only, create the key pair:

    python tools\sign_release.py keygen

    It asks for a passphrase, saves the private key (encrypted with it) to
    %USERPROFILE%\.platescanner\release-signing-key.pem, and prints the public
    key. Paste that into RELEASE_PUBLIC_KEY in platescanner\updates.py and
    commit it. Back up the .pem file and the passphrase somewhere safe (e.g. a
    USB drive): without them no update can be signed again.

For every release, after the release workflow has published it:

    1. Download PlateScanner-Setup.exe from the GitHub release.
    2. python tools\sign_release.py sign PlateScanner-Setup.exe --tag v1.2.0
       (writes PlateScanner-Setup.exe.sig next to it)
    3. Upload PlateScanner-Setup.exe.sig to the same GitHub release.

Until the .sig is uploaded, scanners show the update but only open the release page.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import updates  # noqa: E402

DEFAULT_KEY = Path.home() / ".platescanner" / "release-signing-key.pem"


def public_key_b64(private: Ed25519PrivateKey) -> str:
    raw = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def keygen(key_path: Path, passphrase: str) -> str:
    """Create the key pair; returns the public key to paste into updates.py."""
    if key_path.exists():
        raise SystemExit(f"{key_path} already exists. Delete it first only if you really mean to replace the "
                         "key: scanners with the old public key would then refuse every new update.")
    private = Ed25519PrivateKey.generate()
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.BestAvailableEncryption(passphrase.encode("utf-8")))
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(pem)
    return public_key_b64(private)


def sign(installer: Path, tag: str, key_path: Path, passphrase: str) -> Path:
    """Write <installer>.sig; returns its path."""
    private = serialization.load_pem_private_key(key_path.read_bytes(), passphrase.encode("utf-8"))
    if not isinstance(private, Ed25519PrivateKey):
        raise SystemExit(f"{key_path} is not an Ed25519 key")
    digest = hashlib.sha256(installer.read_bytes()).hexdigest()
    signature = base64.b64encode(private.sign(updates.signed_message(tag, digest)))
    out = installer.with_name(installer.name + ".sig")
    out.write_bytes(signature + b"\n")
    if updates.RELEASE_PUBLIC_KEY and updates.RELEASE_PUBLIC_KEY != public_key_b64(private):
        print("WARNING: this key does not match RELEASE_PUBLIC_KEY in platescanner/updates.py; "
              "scanners will refuse this update.", file=sys.stderr)
    assert updates.verify_signature(tag, digest, signature, public_key_b64(private))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen", help="create the release key pair (once)")
    k.add_argument("--key", type=Path, default=DEFAULT_KEY)
    s = sub.add_parser("sign", help="sign PlateScanner-Setup.exe for one release")
    s.add_argument("installer", type=Path)
    s.add_argument("--tag", required=True, help="the GitHub release tag, e.g. v1.2.0")
    s.add_argument("--key", type=Path, default=DEFAULT_KEY)
    args = ap.parse_args()

    if args.cmd == "keygen":
        pw = getpass.getpass("New passphrase for the signing key: ")
        if len(pw) < 8 or pw != getpass.getpass("Repeat it: "):
            raise SystemExit("Passphrases differ or are shorter than 8 characters.")
        pub = keygen(args.key, pw)
        print(f"Private key saved to {args.key} (back it up, with its passphrase).")
        print("Put this line in platescanner\\updates.py and commit it:\n")
        print(f'RELEASE_PUBLIC_KEY = "{pub}"')
    else:
        if updates.parse_version(args.tag) is None:
            raise SystemExit(f"{args.tag!r} is not a release tag like v1.2.0")
        out = sign(args.installer, args.tag, args.key, getpass.getpass("Signing key passphrase: "))
        print(f"Wrote {out}. Upload it to the {args.tag} release on GitHub.")


if __name__ == "__main__":
    main()
