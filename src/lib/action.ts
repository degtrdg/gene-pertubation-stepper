"use server";

import { promises as fs } from "fs";
import { z } from "zod";
import { proteinNames } from "./protein_names";

export async function getNames(query: string) {
  "use server";

  const searchTerm = query || "";
  const filteredProteins = searchTerm
    ? proteinNames.filter((name) =>
        name.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : proteinNames.slice(0, 10);

  return filteredProteins;
}

export async function submitForm(state: { message: string }, data: FormData) {
  "use server";

  const geneName = data.get("geneName");
  const proteinNames = await getNames(
    geneName instanceof File ? geneName.name : geneName || ""
  );
  const schema = z.object({
    geneName: z
      .string()
      .trim()
      .refine(
        (val) => {
          // Convert both the input and the list items to the same case (e.g., lowercase) for comparison
          const lowerCaseVal = val.toLowerCase();
          return proteinNames
            .map((name) => name.toLowerCase())
            .includes(lowerCaseVal);
        },
        {
          message: "Gene name must be in the protein names list",
        }
      ),
    perturbation: z.string(),
    targetCondition: z.string(),
  });
  const parsedData = schema.safeParse(Object.fromEntries(data));
  if (!parsedData.success) {
    // Return the first error message from the Zod error formatting
    const errorMessage = parsedData.error.issues
      .map((issue) => issue.message)
      .join(", ");
    return { message: errorMessage };
  }
  const goodData = parsedData.data;

  return { message: `success: ${JSON.stringify(goodData)}`, data: goodData };
}
