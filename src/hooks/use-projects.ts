"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ProjectDTO } from "@/types";
import { ApiError, apiFetch } from "@/types/api";

export interface UseProjectsResult {
  projects: ProjectDTO[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  create: (data: {
    name: string;
    description?: string | null;
    instructions?: string | null;
  }) => Promise<ProjectDTO>;
  update: (
    id: string,
    patch: {
      name?: string;
      description?: string | null;
      instructions?: string | null;
    },
  ) => Promise<ProjectDTO>;
  remove: (id: string) => Promise<void>;
}

export function useProjects(): UseProjectsResult {
  const [projects, setProjects] = useState<ProjectDTO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  const refresh = useCallback(() => {
    return apiFetch<{ items: ProjectDTO[] }>("/api/projects")
      .then((data) => {
        if (!mounted.current) return;
        setProjects(data.items);
        setLoading(false);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!mounted.current) return;
        setLoading(false);
        if (cause instanceof ApiError && cause.status === 401) {
          setProjects([]);
          return;
        }
        setError(cause instanceof Error ? cause.message : String(cause));
      });
  }, []);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    return () => {
      mounted.current = false;
    };
  }, [refresh]);

  const create = useCallback(
    async (data: {
      name: string;
      description?: string | null;
      instructions?: string | null;
    }) => {
      const created = await apiFetch<ProjectDTO>("/api/projects", {
        method: "POST",
        body: JSON.stringify(data),
      });
      setProjects((current) => [created, ...current]);
      return created;
    },
    [],
  );

  const update = useCallback(
    async (
      id: string,
      patch: {
        name?: string;
        description?: string | null;
        instructions?: string | null;
      },
    ) => {
      const updated = await apiFetch<ProjectDTO>(`/api/projects/${id}`, {
        method: "PATCH",
        body: JSON.stringify(patch),
      });
      setProjects((current) =>
        current.map((p) => (p.id === id ? { ...p, ...updated } : p)),
      );
      return updated;
    },
    [],
  );

  const remove = useCallback(async (id: string) => {
    await apiFetch<void>(`/api/projects/${id}`, { method: "DELETE" });
    setProjects((current) => current.filter((p) => p.id !== id));
  }, []);

  return { projects, loading, error, refresh, create, update, remove };
}
