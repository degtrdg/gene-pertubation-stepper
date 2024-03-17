"use server";
import MainGraphPage from "@/components/MainGraphPage";
import { getNames } from "@/lib/action";

export default async function Page({
  searchParams,
}: {
  searchParams: {
    query: string;
  };
}) {
  const query = searchParams.query || "";
  const proteinNames = await getNames(query);

  // return <MainGraphPage proteinNames={proteinNames} />;
  return <div>test</div>;
}
