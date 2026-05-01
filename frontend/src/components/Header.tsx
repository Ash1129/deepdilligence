import { Link } from "@tanstack/react-router";
import { Activity } from "lucide-react";

export function Header({ subtle = false }: { subtle?: boolean }) {
  return (
    <header
      className={
        subtle
          ? "sticky top-0 z-40 bg-gradient-to-b from-background/25 via-background/8 to-transparent"
          : "sticky top-0 z-40 bg-gradient-to-b from-background/70 via-background/30 to-transparent backdrop-blur-sm"
      }
    >
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
        <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight text-white drop-shadow-[0_1px_8px_rgba(0,0,0,0.6)]">
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Activity className="h-4 w-4" />
          </span>
          <span>DeepDiligence</span>
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          <Link
            to="/portfolio"
            className="rounded-md px-3 py-1.5 text-white/80 drop-shadow-[0_1px_8px_rgba(0,0,0,0.6)] hover:bg-white/10 hover:text-white"
            activeProps={{ className: "rounded-md px-3 py-1.5 bg-white/10 text-white" }}
          >
            Portfolio
          </Link>
          <Link
            to="/weekly"
            className="rounded-md px-3 py-1.5 text-white/80 drop-shadow-[0_1px_8px_rgba(0,0,0,0.6)] hover:bg-white/10 hover:text-white"
            activeProps={{ className: "rounded-md px-3 py-1.5 bg-white/10 text-white" }}
          >
            Weekly Rankings
          </Link>
          <Link
            to="/analyze"
            className="rounded-md px-3 py-1.5 text-white/80 drop-shadow-[0_1px_8px_rgba(0,0,0,0.6)] hover:bg-white/10 hover:text-white"
            activeProps={{ className: "rounded-md px-3 py-1.5 bg-white/10 text-white" }}
          >
            Analyse
          </Link>
        </nav>
      </div>
    </header>
  );
}