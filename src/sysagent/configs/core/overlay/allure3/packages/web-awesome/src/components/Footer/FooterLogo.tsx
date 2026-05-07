import { getReportOptions } from "@allurereport/web-commons";
import { Text } from "@allurereport/web-components";
import type { AwesomeReportOptions } from "types";
import * as styles from "./styles.scss";

export const FooterLogo = () => {
  const { reportName } = getReportOptions<AwesomeReportOptions>() ?? {};

  return (
    <div className={styles["footer-logo"]}>
      <Text type="paragraph" size="m" bold className={styles["footer-logo"]}>
        {reportName || "Allure Report"}
      </Text>
    </div>
  );
};
