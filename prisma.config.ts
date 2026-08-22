import "dotenv/config";
import { defineConfig } from "prisma/config";

export default defineConfig({
  schema: "prisma/schema.prisma",
  migrations: {
    path: "prisma/migrations",
  },
  engine: "classic",
  datasource: {
    url:
      process.env.DATABASE_URL?.trim() ||
      "postgresql://bravien:bravien@localhost:5432/bravien?schema=public",
  },
});
