import { PrismaClient } from "@prisma/client";
import { hash } from "bcryptjs";

const prisma = new PrismaClient();

async function main() {
  const adminEmail = (process.env.SEED_ADMIN_EMAIL || "admin@bravien.local").toLowerCase().trim();
  const adminPassword = process.env.SEED_ADMIN_PASSWORD || "admin123456";
  const passwordHash = await hash(adminPassword, 12);

  const admin = await prisma.user.upsert({
    where: { email: adminEmail },
    update: {
      passwordHash,
      role: "ADMIN",
    },
    create: {
      email: adminEmail,
      name: "Bravien Admin",
      passwordHash,
      role: "ADMIN",
    },
  });

  console.log(`Seeded admin user: ${admin.email} (ID: ${admin.id})`);
}

main()
  .catch((e) => {
    console.error("Error during seeding:", e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
